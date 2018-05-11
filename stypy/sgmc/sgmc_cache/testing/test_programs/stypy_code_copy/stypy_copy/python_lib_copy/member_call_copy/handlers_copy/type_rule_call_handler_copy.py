
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import os
2: import sys
3: import inspect
4: 
5: from call_handler_copy import CallHandler
6: from ....python_lib_copy.python_types_copy import type_inference_copy
7: from .... import stypy_parameters_copy
8: from ....python_lib_copy.type_rules_copy.type_groups_copy.type_groups_copy import *
9: from ....python_lib_copy.type_rules_copy.type_groups_copy.type_group_copy import BaseTypeGroup
10: 
11: 
12: class TypeRuleCallHandler(CallHandler):
13:     '''
14:     This call handler uses type rule files (Python files with a special structure) to determine acceptable parameters
15:     and return types for the calls of a certain module/class and its callable members. The handler dynamically search,
16:     load and use these rule files to resolve calls.
17:     '''
18: 
19:     # Cache of found rule files
20:     type_rule_cache = dict()
21: 
22:     # Cache of not found rule files (to improve performance)
23:     unavailable_type_rule_cache = dict()
24: 
25:     @staticmethod
26:     def __rule_files(parent_name, entity_name):
27:         '''
28:         For a call to parent_name.entity_name(...), compose the name of the type rule file that will correspond to the
29:         entity or its parent, to look inside any of them for suitable rules to apply
30:         :param parent_name: Parent entity (module/class) name
31:         :param entity_name: Callable entity (function/method) name
32:         :return: A tuple of (name of the rule file of the parent, name of the type rule of the entity)
33:         '''
34:         parent_type_rule_file = stypy_parameters_copy.ROOT_PATH + stypy_parameters_copy.RULE_FILE_PATH + parent_name + "/" \
35:                                 + parent_name + stypy_parameters_copy.type_rule_file_postfix + ".py"
36: 
37:         own_type_rule_file = stypy_parameters_copy.ROOT_PATH + stypy_parameters_copy.RULE_FILE_PATH + parent_name + "/" \
38:                              + entity_name.split('.')[-1] + "/" + entity_name.split('.')[
39:                                  -1] + stypy_parameters_copy.type_rule_file_postfix + ".py"
40: 
41:         return parent_type_rule_file, own_type_rule_file
42: 
43:     @staticmethod
44:     def __dependent_type_in_rule_params(params):
45:         '''
46:         Check if a list of params has dependent types: Types that have to be called somewhat in order to obtain the
47:         real type they represent.
48:         :param params: List of types
49:         :return: bool
50:         '''
51:         return len(filter(lambda par: isinstance(par, DependentType), params)) > 0
52: 
53:     @staticmethod
54:     def __has_varargs_in_rule_params(params):
55:         '''
56:         Check if a list of params has variable number of arguments
57:         :param params: List of types
58:         :return: bool
59:         '''
60:         return len(filter(lambda par: isinstance(par, VarArgType), params)) > 0
61: 
62:     @staticmethod
63:     def __get_arguments(argument_tuple, current_pos, rule_arity):
64:         '''
65:         Obtain a list composed by the arguments present in argument_tuple, except the one in current_pos limited
66:         to rule_arity size. This is used when invoking dependent rules
67:         :param argument_tuple:
68:         :param current_pos:
69:         :param rule_arity:
70:         :return:
71:         '''
72:         if rule_arity == 0:
73:             return []
74: 
75:         temp_list = []
76:         for i in range(len(argument_tuple)):
77:             if not i == current_pos:
78:                 temp_list.append(argument_tuple[i])
79: 
80:         return tuple(temp_list[0:rule_arity])
81: 
82:     def invoke_dependent_rules(self, localization, rule_params, arguments):
83:         '''
84:         As we said, some rules may contain special types called DependentTypes. These types have to be invoked in
85:         order to check that the rule matches with the call or other necessary operations. Dependent types may have
86:         several forms, and are called with all the arguments that are checked against the type rule except the one
87:         that matches de dependent type, limited by the Dependent type declared rule arity. For example a Dependent
88:         Type may be defined like this (see type_groups.py for all the Dependent types defined):
89: 
90:         Overloads__eq__ = HasMember("__eq__", DynamicType, 1)
91: 
92:         This means that Overloads__eq__ matches with all the objects that has a method named __eq__ that has no
93:         predefined return type and an arity of 1 parameter. On the other hand, a type rule may be defined like this:
94: 
95:         ((Overloads__eq__, AnyType), DynamicType)
96: 
97:         This means that the type rule matches with a call that has a first argument which overloads the method
98:         __eq__ and any kind of second arguments. Although __eq__ is a method that should return bool (is the ==
99:         operator) this is not compulsory in Python, the __eq__ method may return anything and this anything will be
100:         the result of the rule. So we have to call __eq__ with the second argument (all the arguments but the one
101:         that matches with the DependentType limited to the declared dependent type arity), capture and return the
102:         result. This is basically the functionality of this method.
103: 
104:         Note that invocation to a method means that the type rule call handler (or another one) may be used again
105:         against the invoked method (__eq__ in our example).
106: 
107:         :param localization: Caller information
108:         :param rule_params: Rule file entry
109:         :param arguments: Arguments passed to the call that matches against the rule file.
110:         :return:
111:         '''
112:         temp_rule = []
113:         needs_reevaluation = False
114:         for i in range(len(rule_params)):
115:             # Are we dealing with a dependent type?
116:             if isinstance(rule_params[i], DependentType):
117:                 # Invoke it with the parameters we described previously
118:                 correct_invokation, equivalent_type = rule_params[i](
119:                     localization, *self.__get_arguments(arguments, i, rule_params[i].call_arity))
120: 
121:                 # Is the invocation correct?
122:                 if not correct_invokation:
123:                     # No, return that this rule do not really match
124:                     return False, None, needs_reevaluation, None
125:                 else:
126:                     # The equivalent type is the one determined by the dependent type rule invocation
127:                     if equivalent_type is not None:
128:                         # By convention, if the declared rule result is UndefinedType, the call will be reevaluated
129:                         # substituting the dependent type in position i with its equivalent_type
130:                         if rule_params[i].expected_return_type is UndefinedType:
131:                             needs_reevaluation = True
132:                             temp_rule.append(equivalent_type)
133:                         # By convention, if the declared rule result is DynamicType, it is substituted by its equivalent
134:                         # type. This is the most common case
135:                         if rule_params[i].expected_return_type is DynamicType:
136:                             return True, None, needs_reevaluation, equivalent_type
137:                         # #TO DO: This fails
138:                         # if rule_params[i].expected_return_type is equivalent_type.get_python_type():
139:                         #     needs_reevaluation = True
140:                         #     temp_rule.append(equivalent_type)
141:                         # else:
142:                         #     return False, None, needs_reevaluation, None
143: 
144:                         # Some dependent types have a declared fixed return type (not like our previous example, which
145:                         # has DynamicType instead. In that case, if the dependent type invocation do not return the
146:                         # expected type, this means that the match is not valid and another rule has to be used to
147:                         # resolve the call.
148:                         if rule_params[i].expected_return_type is not equivalent_type.get_python_type():
149:                             return False, None, needs_reevaluation, None
150:                     else:
151:                         temp_rule.append(rule_params[i])
152:             else:
153:                 temp_rule.append(rule_params[i])
154:         return True, tuple(temp_rule), needs_reevaluation, None
155: 
156:     # TODO: Remove?
157:     # def get_rule_files(self, proxy_obj, callable_entity):
158:     #     '''
159:     #     Obtain the corresponding rule files to the callable entity, using its name and its containers name.
160:     #     :param proxy_obj: TypeInferenceProxy that holds the callable_entity
161:     #     :param callable_entity: Python callable_entity
162:     #     :return:
163:     #     '''
164:     #     if inspect.ismethod(callable_entity) or inspect.ismethoddescriptor(callable_entity):
165:     #         parent_type_rule_file, own_type_rule_file = self.__rule_files(proxy_obj.parent_proxy.parent_proxy.name,
166:     #                                                                       proxy_obj.parent_proxy.name)
167:     #     else:
168:     #         parent_type_rule_file, own_type_rule_file = self.__rule_files(proxy_obj.parent_proxy.name, proxy_obj.name)
169:     #
170:     #     parent_exist = os.path.isfile(parent_type_rule_file)
171:     #     own_exist = os.path.isfile(own_type_rule_file)
172:     #
173:     #     return parent_exist, own_exist, parent_type_rule_file, own_type_rule_file
174: 
175:     def applies_to(self, proxy_obj, callable_entity):
176:         '''
177:         This method determines if this call handler is able to respond to a call to callable_entity. The call handler
178:         respond to any callable code that has a rule file associated. This method search the rule file and, if found,
179:         loads and caches it for performance reasons. Cache also allows us to not to look for the same file on the
180:         hard disk over and over, saving much time. callable_entity rule files have priority over the rule files of
181:         their parent entity should both exist.
182: 
183:         :param proxy_obj: TypeInferenceProxy that hold the callable entity
184:         :param callable_entity: Callable entity
185:         :return: bool
186:         '''
187:         # We have a class, calling a class means instantiating it
188:         if inspect.isclass(callable_entity):
189:             cache_name = proxy_obj.name + ".__init__"
190:         else:
191:             cache_name = proxy_obj.name
192: 
193:         # No rule file for this callable (from the cache)
194:         if self.unavailable_type_rule_cache.get(cache_name, False):
195:             return False
196: 
197:         # There are a rule file for this callable (from the cache)
198:         if self.type_rule_cache.get(cache_name, False):
199:             return True
200: 
201:         # There are a rule file for this callable parent entity (from the cache)
202:         if proxy_obj.parent_proxy is not None:
203:             if self.type_rule_cache.get(proxy_obj.parent_proxy.name, False):
204:                 return True
205: 
206:         # TODO: Remove?
207:         # if proxy_obj.name in self.unavailable_type_rule_cache:
208:         #     return False
209:         #
210:         # if proxy_obj.name in self.type_rule_cache:
211:         #     return True
212: 
213:         # if proxy_obj.parent_proxy is not None:
214:         #     if proxy_obj.parent_proxy.name in self.type_rule_cache:
215:         #         return True
216: 
217:         # Obtain available rule files depending on the type of entity that is going to be called
218:         if inspect.ismethod(callable_entity) or inspect.ismethoddescriptor(callable_entity) or (
219:                     inspect.isbuiltin(callable_entity) and
220:                     (inspect.isclass(proxy_obj.parent_proxy.get_python_entity()))):
221:             try:
222:                 parent_type_rule_file, own_type_rule_file = self.__rule_files(callable_entity.__objclass__.__module__,
223:                                                                               callable_entity.__objclass__.__name__,
224:                                                                               )
225:             except:
226:                 if inspect.ismodule(proxy_obj.parent_proxy.get_python_entity()):
227:                     parent_type_rule_file, own_type_rule_file = self.__rule_files(
228:                         proxy_obj.parent_proxy.name,
229:                         proxy_obj.parent_proxy.name)
230:                 else:
231:                     parent_type_rule_file, own_type_rule_file = self.__rule_files(
232:                         proxy_obj.parent_proxy.parent_proxy.name,
233:                         proxy_obj.parent_proxy.name)
234:         else:
235:             parent_type_rule_file, own_type_rule_file = self.__rule_files(proxy_obj.parent_proxy.name, proxy_obj.name)
236: 
237:         # Determine which rule file to use
238:         parent_exist = os.path.isfile(parent_type_rule_file)
239:         own_exist = os.path.isfile(own_type_rule_file)
240:         file_path = ""
241: 
242:         if parent_exist:
243:             file_path = parent_type_rule_file
244: 
245:         if own_exist:
246:             file_path = own_type_rule_file
247: 
248:         # Load rule file
249:         if parent_exist or own_exist:
250:             dirname = os.path.dirname(file_path)
251:             file_ = file_path.split('/')[-1][0:-3]
252: 
253:             sys.path.append(dirname)
254:             module = __import__(file_, globals(), locals())
255:             entity_name = proxy_obj.name.split('.')[-1]
256:             try:
257:                 # Is there a rule for the specific entity even if the container of the entity has a rule file?
258:                 # This way rule files are used while they are created. All rule files declare a member called
259:                 # type_rules_of_members
260:                 rules = module.type_rules_of_members[entity_name]
261: 
262:                 # Dynamically load-time calculated rules (unused yet)
263:                 if inspect.isfunction(rules):
264:                     rules = rules()  # rules(entity_name)
265: 
266:                 # Cache loaded rules for the member
267:                 self.type_rule_cache[cache_name] = rules
268:             except:
269:                 # Cache unexisting rules for the member
270:                 self.unavailable_type_rule_cache[cache_name] = True
271:                 return False
272: 
273:         if not (parent_exist or own_exist):
274:             if proxy_obj.name not in self.unavailable_type_rule_cache:
275:                 # Cache unexisting rules for the member
276:                 self.unavailable_type_rule_cache[cache_name] = True
277: 
278:         return parent_exist or own_exist
279: 
280:     def __get_rules_and_name(self, entity_name, parent_name):
281:         '''
282:         Obtain a member name and its type rules
283:         :param entity_name: Entity name
284:         :param parent_name: Entity container name
285:         :return: tuple (name, rules tied to this name)
286:         '''
287:         if entity_name in self.type_rule_cache:
288:             name = entity_name
289:             rules = self.type_rule_cache[entity_name]
290: 
291:             return name, rules
292: 
293:         if parent_name in self.type_rule_cache:
294:             name = parent_name
295:             rules = self.type_rule_cache[parent_name]
296: 
297:             return name, rules
298: 
299:     @staticmethod
300:     def __format_admitted_params(name, rules, arguments, call_arity):
301:         '''
302:         Pretty-print error message when no type rule for the member matches with the arguments of the call
303:         :param name: Member name
304:         :param rules: Rules tied to this member name
305:         :param arguments: Call arguments
306:         :param call_arity: Call arity
307:         :return:
308:         '''
309:         params_strs = [""] * call_arity
310:         first_rule = True
311:         arities = []
312: 
313:         # Problem with argument number?
314:         rules_with_enough_arguments = False
315:         for (params_in_rules, return_type) in rules:
316:             rule_len = len(params_in_rules)
317:             if rule_len not in arities:
318:                 arities.append(rule_len)
319: 
320:             if len(params_in_rules) == call_arity:
321:                 rules_with_enough_arguments = True
322: 
323:         if not rules_with_enough_arguments:
324:             str_arities = ""
325:             for i in range(len(arities)):
326:                 str_arities += str(arities[i])
327:                 if len(arities) > 1:
328:                     if i == len(arities) - 1:
329:                         str_arities += " or "
330:                     else:
331:                         str_arities += ", "
332:             return "The invocation was performed with {0} argument(s), but only {1} argument(s) are accepted".format(
333:                 call_arity,
334:                 str_arities)
335: 
336:         for (params_in_rules, return_type) in rules:
337:             if len(params_in_rules) == call_arity:
338:                 for i in range(call_arity):
339:                     value = str(params_in_rules[i])
340:                     if value not in params_strs[i]:
341:                         if not first_rule:
342:                             params_strs[i] += " \/ "
343:                         params_strs[i] += value
344: 
345:                 first_rule = False
346: 
347:         repr_ = ""
348:         for str_ in params_strs:
349:             repr_ += str_ + ", "
350: 
351:         return name + "(" + repr_[:-2] + ") expected"
352: 
353:     @staticmethod
354:     def __compare(params_in_rules, argument_types):
355:         '''
356:         Most important function in the call handler, determines if a rule matches with the call arguments initially
357:         (this means that the rule can potentially match with the argument types because the structure of the arguments,
358:         but if the rule has dependent types, this match could not be so in the end, once the dependent types are
359:         evaluated.
360:         :param params_in_rules: Parameters declared on the rule
361:         :param argument_types: Types passed on the call
362:         :return:
363:         '''
364:         for i in range(len(params_in_rules)):
365:             param = params_in_rules[i]
366:             # Always should be declared at the end of the rule list for a member, so no more iterations should occur
367:             if isinstance(param, VarArgType):
368:                 continue
369:             # Type group: An special entity that matches against several Python types (it overloads its __eq__ method)
370:             if isinstance(param, BaseTypeGroup):
371:                 if not param == argument_types[i]:
372:                     return False
373:             else:
374:                 # Match against raw Python types
375:                 if not param == argument_types[i].get_python_type():
376:                     return False
377: 
378:         return True
379: 
380:     @staticmethod
381:     def __create_return_type(localization, ret_type, argument_types):
382:         '''
383:         Create a suitable return type for the rule (if the return type is a dependent type, this invoked it against
384:         the call arguments to obtain it)
385:         :param localization: Caller information
386:         :param ret_type: Declared return type in a matched rule
387:         :param argument_types: Arguments of the call
388:         :return:
389:         '''
390:         if isinstance(ret_type, DependentType):
391:             return type_inference_copy.type_inference_proxy.TypeInferenceProxy.instance(
392:                 ret_type(localization, argument_types))
393:         else:
394:             return type_inference_copy.type_inference_proxy.TypeInferenceProxy.instance(
395:                 ret_type)
396: 
397:     def get_parameter_arity(self, proxy_obj, callable_entity):
398:         '''
399:         Obtain the minimum and maximum arity of a callable element using the type rules declared for it. It also
400:         indicates if it has varargs (infinite arity)
401:         :param proxy_obj: TypeInferenceProxy that holds the callable entity
402:         :param callable_entity: Callable entity
403:         :return: list of possible arities, bool (wether it has varargs or not)
404:         '''
405:         if inspect.isclass(callable_entity):
406:             cache_name = proxy_obj.name + ".__init__"
407:         else:
408:             cache_name = proxy_obj.name
409: 
410:         has_varargs = False
411:         arities = []
412:         name, rules = self.__get_rules_and_name(cache_name, proxy_obj.parent_proxy.name)
413:         for (params_in_rules, return_type) in rules:
414:             if self.__has_varargs_in_rule_params(params_in_rules):
415:                 has_varargs = True
416:             num = len(params_in_rules)
417:             if num not in arities:
418:                 arities.append(num)
419: 
420:         return arities, has_varargs
421: 
422:     def __call__(self, proxy_obj, localization, callable_entity, *arg_types, **kwargs_types):
423:         '''
424:         Calls the callable entity with its type rules to determine its return type.
425: 
426:         :param proxy_obj: TypeInferenceProxy that hold the callable entity
427:         :param localization: Caller information
428:         :param callable_entity: Callable entity
429:         :param arg_types: Arguments
430:         :param kwargs_types: Keyword arguments
431:         :return: Return type of the call
432:         '''
433:         if inspect.isclass(callable_entity):
434:             cache_name = proxy_obj.name + ".__init__"
435:         else:
436:             cache_name = proxy_obj.name
437: 
438:         name, rules = self.__get_rules_and_name(cache_name, proxy_obj.parent_proxy.name)
439: 
440:         argument_types = None
441: 
442:         # If there is only one rule, we transfer to the rule the ability of reporting its errors more precisely
443:         if len(rules) > 1:
444:             prints_msg = True
445:         else:
446:             prints_msg = False
447: 
448:         # Method?
449:         if inspect.ismethod(callable_entity) or inspect.ismethoddescriptor(callable_entity):
450:             # Are we calling with a type variable instead with a type instance?
451:             if not proxy_obj.parent_proxy.is_type_instance():
452:                 # Is the first parameter a subtype of the type variable used to perform the call?
453:                 if not issubclass(arg_types[0].python_entity, callable_entity.__objclass__):
454:                     # No: Report a suitable error
455:                     argument_types = tuple(list(arg_types) + kwargs_types.values())
456:                     usage_hint = self.__format_admitted_params(name, rules, argument_types, len(argument_types))
457:                     arg_description = str(argument_types)
458:                     arg_description = arg_description.replace(",)", ")")
459:                     return TypeError(localization,
460:                                      "Call to {0}{1} is invalid. Argument 1 requires a '{3}' but received "
461:                                      "a '{4}' \n\t{2}".format(name, arg_description, usage_hint,
462:                                                               str(callable_entity.__objclass__),
463:                                                               str(arg_types[0].python_entity)),
464:                                      prints_msg=prints_msg)
465:                 else:
466:                     argument_types = tuple(list(arg_types[1:]) + kwargs_types.values())
467: 
468:         # Argument types passed for the call (if not previously initialized)
469:         if argument_types is None:
470:             argument_types = tuple(list(arg_types))  # + kwargs_types.values())
471: 
472:         call_arity = len(argument_types)
473: 
474:         # Examine each rule corresponding to this member
475:         for (params_in_rules, return_type) in rules:
476:             # Discard rules that do not match arity
477:             if len(params_in_rules) == call_arity or self.__has_varargs_in_rule_params(params_in_rules):
478:                 # The passed arguments matches with one of the rules
479:                 if self.__compare(params_in_rules, argument_types):
480:                     # Is there a dependent type on the matched rule?
481:                     if self.__dependent_type_in_rule_params(params_in_rules):
482:                         # We obtain the equivalent type rule of the matched rule (calculated during comparison)
483:                         correct, equivalent_rule, needs_reevaluation, invokation_rt = self.invoke_dependent_rules(
484:                             localization, params_in_rules, argument_types)
485:                         # Errors in dependent type invocation?
486:                         if correct:
487:                             # Correct call, return the rule declared type
488:                             if not needs_reevaluation:
489:                                 # The rule says that this is the type to be returned, as it couldn't be predetermined
490:                                 # in the rule
491:                                 if invokation_rt is not None:
492:                                     return self.__create_return_type(localization, invokation_rt, argument_types)
493:                                     # return type_inference.type_inference_proxy.TypeInferenceProxy.instance(
494:                                     #     invokation_rt)
495: 
496:                                 # Comprobacion de Dependent return type
497:                                 return self.__create_return_type(localization, return_type, argument_types)
498:                                 # return type_inference.type_inference_proxy.TypeInferenceProxy.instance(return_type)
499:                             else:
500:                                 # As one of the dependent rules has a non-predefined return type, we need to obtain it
501:                                 # and evaluate it again
502:                                 for (params_in_rules2, return_type2) in rules:
503:                                     # The passed arguments matches with one of the rules
504:                                     if params_in_rules2 == equivalent_rule:
505:                                         # Create the return type
506:                                         return self.__create_return_type(localization, return_type2, argument_types)
507:                     else:
508:                         # Create the return type
509:                         return self.__create_return_type(localization, return_type, argument_types)
510: 
511:         # No rule is matched, return error
512:         usage_hint = self.__format_admitted_params(name, rules, argument_types, len(argument_types))
513: 
514:         arg_description = str(argument_types)
515:         arg_description = arg_description.replace(",)", ")")
516:         return TypeError(localization, "Call to {0}{1} is invalid.\n\t{2}".format(name, arg_description,
517:                                                                                   usage_hint), prints_msg=prints_msg)
518: 

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

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'import sys' statement (line 2)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import inspect' statement (line 3)
import inspect

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'inspect', inspect, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from call_handler_copy import CallHandler' statement (line 5)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/handlers_copy/')
import_6198 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'call_handler_copy')

if (type(import_6198) is not StypyTypeError):

    if (import_6198 != 'pyd_module'):
        __import__(import_6198)
        sys_modules_6199 = sys.modules[import_6198]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'call_handler_copy', sys_modules_6199.module_type_store, module_type_store, ['CallHandler'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_6199, sys_modules_6199.module_type_store, module_type_store)
    else:
        from call_handler_copy import CallHandler

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'call_handler_copy', None, module_type_store, ['CallHandler'], [CallHandler])

else:
    # Assigning a type to the variable 'call_handler_copy' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'call_handler_copy', import_6198)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/handlers_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy import type_inference_copy' statement (line 6)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/handlers_copy/')
import_6200 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy')

if (type(import_6200) is not StypyTypeError):

    if (import_6200 != 'pyd_module'):
        __import__(import_6200)
        sys_modules_6201 = sys.modules[import_6200]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy', sys_modules_6201.module_type_store, module_type_store, ['type_inference_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_6201, sys_modules_6201.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy import type_inference_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy', None, module_type_store, ['type_inference_copy'], [type_inference_copy])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy', import_6200)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/handlers_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy import stypy_parameters_copy' statement (line 7)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/handlers_copy/')
import_6202 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'testing.test_programs.stypy_code_copy.stypy_copy')

if (type(import_6202) is not StypyTypeError):

    if (import_6202 != 'pyd_module'):
        __import__(import_6202)
        sys_modules_6203 = sys.modules[import_6202]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'testing.test_programs.stypy_code_copy.stypy_copy', sys_modules_6203.module_type_store, module_type_store, ['stypy_parameters_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_6203, sys_modules_6203.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy import stypy_parameters_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'testing.test_programs.stypy_code_copy.stypy_copy', None, module_type_store, ['stypy_parameters_copy'], [stypy_parameters_copy])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'testing.test_programs.stypy_code_copy.stypy_copy', import_6202)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/handlers_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.type_rules_copy.type_groups_copy.type_groups_copy import ' statement (line 8)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/handlers_copy/')
import_6204 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.type_rules_copy.type_groups_copy.type_groups_copy')

if (type(import_6204) is not StypyTypeError):

    if (import_6204 != 'pyd_module'):
        __import__(import_6204)
        sys_modules_6205 = sys.modules[import_6204]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.type_rules_copy.type_groups_copy.type_groups_copy', sys_modules_6205.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_6205, sys_modules_6205.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.type_rules_copy.type_groups_copy.type_groups_copy import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.type_rules_copy.type_groups_copy.type_groups_copy', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.type_rules_copy.type_groups_copy.type_groups_copy' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.type_rules_copy.type_groups_copy.type_groups_copy', import_6204)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/handlers_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.type_rules_copy.type_groups_copy.type_group_copy import BaseTypeGroup' statement (line 9)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/handlers_copy/')
import_6206 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.type_rules_copy.type_groups_copy.type_group_copy')

if (type(import_6206) is not StypyTypeError):

    if (import_6206 != 'pyd_module'):
        __import__(import_6206)
        sys_modules_6207 = sys.modules[import_6206]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.type_rules_copy.type_groups_copy.type_group_copy', sys_modules_6207.module_type_store, module_type_store, ['BaseTypeGroup'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_6207, sys_modules_6207.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.type_rules_copy.type_groups_copy.type_group_copy import BaseTypeGroup

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.type_rules_copy.type_groups_copy.type_group_copy', None, module_type_store, ['BaseTypeGroup'], [BaseTypeGroup])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.type_rules_copy.type_groups_copy.type_group_copy' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.type_rules_copy.type_groups_copy.type_group_copy', import_6206)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/handlers_copy/')

# Declaration of the 'TypeRuleCallHandler' class
# Getting the type of 'CallHandler' (line 12)
CallHandler_6208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 26), 'CallHandler')

class TypeRuleCallHandler(CallHandler_6208, ):
    str_6209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, (-1)), 'str', '\n    This call handler uses type rule files (Python files with a special structure) to determine acceptable parameters\n    and return types for the calls of a certain module/class and its callable members. The handler dynamically search,\n    load and use these rule files to resolve calls.\n    ')
    
    # Assigning a Call to a Name (line 20):
    
    # Assigning a Call to a Name (line 23):

    @staticmethod
    @norecursion
    def __rule_files(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__rule_files'
        module_type_store = module_type_store.open_function_context('__rule_files', 25, 4, False)
        
        # Passed parameters checking function
        TypeRuleCallHandler.__rule_files.__dict__.__setitem__('stypy_localization', localization)
        TypeRuleCallHandler.__rule_files.__dict__.__setitem__('stypy_type_of_self', None)
        TypeRuleCallHandler.__rule_files.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeRuleCallHandler.__rule_files.__dict__.__setitem__('stypy_function_name', '__rule_files')
        TypeRuleCallHandler.__rule_files.__dict__.__setitem__('stypy_param_names_list', ['parent_name', 'entity_name'])
        TypeRuleCallHandler.__rule_files.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeRuleCallHandler.__rule_files.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeRuleCallHandler.__rule_files.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeRuleCallHandler.__rule_files.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeRuleCallHandler.__rule_files.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeRuleCallHandler.__rule_files.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, None, module_type_store, '__rule_files', ['parent_name', 'entity_name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__rule_files', localization, ['entity_name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__rule_files(...)' code ##################

        str_6210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, (-1)), 'str', '\n        For a call to parent_name.entity_name(...), compose the name of the type rule file that will correspond to the\n        entity or its parent, to look inside any of them for suitable rules to apply\n        :param parent_name: Parent entity (module/class) name\n        :param entity_name: Callable entity (function/method) name\n        :return: A tuple of (name of the rule file of the parent, name of the type rule of the entity)\n        ')
        
        # Assigning a BinOp to a Name (line 34):
        
        # Assigning a BinOp to a Name (line 34):
        # Getting the type of 'stypy_parameters_copy' (line 34)
        stypy_parameters_copy_6211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 32), 'stypy_parameters_copy')
        # Obtaining the member 'ROOT_PATH' of a type (line 34)
        ROOT_PATH_6212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 32), stypy_parameters_copy_6211, 'ROOT_PATH')
        # Getting the type of 'stypy_parameters_copy' (line 34)
        stypy_parameters_copy_6213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 66), 'stypy_parameters_copy')
        # Obtaining the member 'RULE_FILE_PATH' of a type (line 34)
        RULE_FILE_PATH_6214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 66), stypy_parameters_copy_6213, 'RULE_FILE_PATH')
        # Applying the binary operator '+' (line 34)
        result_add_6215 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 32), '+', ROOT_PATH_6212, RULE_FILE_PATH_6214)
        
        # Getting the type of 'parent_name' (line 34)
        parent_name_6216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 105), 'parent_name')
        # Applying the binary operator '+' (line 34)
        result_add_6217 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 103), '+', result_add_6215, parent_name_6216)
        
        str_6218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 119), 'str', '/')
        # Applying the binary operator '+' (line 34)
        result_add_6219 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 117), '+', result_add_6217, str_6218)
        
        # Getting the type of 'parent_name' (line 35)
        parent_name_6220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 34), 'parent_name')
        # Applying the binary operator '+' (line 35)
        result_add_6221 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 32), '+', result_add_6219, parent_name_6220)
        
        # Getting the type of 'stypy_parameters_copy' (line 35)
        stypy_parameters_copy_6222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 48), 'stypy_parameters_copy')
        # Obtaining the member 'type_rule_file_postfix' of a type (line 35)
        type_rule_file_postfix_6223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 48), stypy_parameters_copy_6222, 'type_rule_file_postfix')
        # Applying the binary operator '+' (line 35)
        result_add_6224 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 46), '+', result_add_6221, type_rule_file_postfix_6223)
        
        str_6225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 95), 'str', '.py')
        # Applying the binary operator '+' (line 35)
        result_add_6226 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 93), '+', result_add_6224, str_6225)
        
        # Assigning a type to the variable 'parent_type_rule_file' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'parent_type_rule_file', result_add_6226)
        
        # Assigning a BinOp to a Name (line 37):
        
        # Assigning a BinOp to a Name (line 37):
        # Getting the type of 'stypy_parameters_copy' (line 37)
        stypy_parameters_copy_6227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 29), 'stypy_parameters_copy')
        # Obtaining the member 'ROOT_PATH' of a type (line 37)
        ROOT_PATH_6228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 29), stypy_parameters_copy_6227, 'ROOT_PATH')
        # Getting the type of 'stypy_parameters_copy' (line 37)
        stypy_parameters_copy_6229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 63), 'stypy_parameters_copy')
        # Obtaining the member 'RULE_FILE_PATH' of a type (line 37)
        RULE_FILE_PATH_6230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 63), stypy_parameters_copy_6229, 'RULE_FILE_PATH')
        # Applying the binary operator '+' (line 37)
        result_add_6231 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 29), '+', ROOT_PATH_6228, RULE_FILE_PATH_6230)
        
        # Getting the type of 'parent_name' (line 37)
        parent_name_6232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 102), 'parent_name')
        # Applying the binary operator '+' (line 37)
        result_add_6233 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 100), '+', result_add_6231, parent_name_6232)
        
        str_6234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 116), 'str', '/')
        # Applying the binary operator '+' (line 37)
        result_add_6235 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 114), '+', result_add_6233, str_6234)
        
        
        # Obtaining the type of the subscript
        int_6236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 54), 'int')
        
        # Call to split(...): (line 38)
        # Processing the call arguments (line 38)
        str_6239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 49), 'str', '.')
        # Processing the call keyword arguments (line 38)
        kwargs_6240 = {}
        # Getting the type of 'entity_name' (line 38)
        entity_name_6237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 31), 'entity_name', False)
        # Obtaining the member 'split' of a type (line 38)
        split_6238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 31), entity_name_6237, 'split')
        # Calling split(args, kwargs) (line 38)
        split_call_result_6241 = invoke(stypy.reporting.localization.Localization(__file__, 38, 31), split_6238, *[str_6239], **kwargs_6240)
        
        # Obtaining the member '__getitem__' of a type (line 38)
        getitem___6242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 31), split_call_result_6241, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 38)
        subscript_call_result_6243 = invoke(stypy.reporting.localization.Localization(__file__, 38, 31), getitem___6242, int_6236)
        
        # Applying the binary operator '+' (line 38)
        result_add_6244 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 29), '+', result_add_6235, subscript_call_result_6243)
        
        str_6245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 60), 'str', '/')
        # Applying the binary operator '+' (line 38)
        result_add_6246 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 58), '+', result_add_6244, str_6245)
        
        
        # Obtaining the type of the subscript
        int_6247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 33), 'int')
        
        # Call to split(...): (line 38)
        # Processing the call arguments (line 38)
        str_6250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 84), 'str', '.')
        # Processing the call keyword arguments (line 38)
        kwargs_6251 = {}
        # Getting the type of 'entity_name' (line 38)
        entity_name_6248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 66), 'entity_name', False)
        # Obtaining the member 'split' of a type (line 38)
        split_6249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 66), entity_name_6248, 'split')
        # Calling split(args, kwargs) (line 38)
        split_call_result_6252 = invoke(stypy.reporting.localization.Localization(__file__, 38, 66), split_6249, *[str_6250], **kwargs_6251)
        
        # Obtaining the member '__getitem__' of a type (line 38)
        getitem___6253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 66), split_call_result_6252, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 38)
        subscript_call_result_6254 = invoke(stypy.reporting.localization.Localization(__file__, 38, 66), getitem___6253, int_6247)
        
        # Applying the binary operator '+' (line 38)
        result_add_6255 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 64), '+', result_add_6246, subscript_call_result_6254)
        
        # Getting the type of 'stypy_parameters_copy' (line 39)
        stypy_parameters_copy_6256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 39), 'stypy_parameters_copy')
        # Obtaining the member 'type_rule_file_postfix' of a type (line 39)
        type_rule_file_postfix_6257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 39), stypy_parameters_copy_6256, 'type_rule_file_postfix')
        # Applying the binary operator '+' (line 39)
        result_add_6258 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 37), '+', result_add_6255, type_rule_file_postfix_6257)
        
        str_6259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 86), 'str', '.py')
        # Applying the binary operator '+' (line 39)
        result_add_6260 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 84), '+', result_add_6258, str_6259)
        
        # Assigning a type to the variable 'own_type_rule_file' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'own_type_rule_file', result_add_6260)
        
        # Obtaining an instance of the builtin type 'tuple' (line 41)
        tuple_6261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 41)
        # Adding element type (line 41)
        # Getting the type of 'parent_type_rule_file' (line 41)
        parent_type_rule_file_6262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 15), 'parent_type_rule_file')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 15), tuple_6261, parent_type_rule_file_6262)
        # Adding element type (line 41)
        # Getting the type of 'own_type_rule_file' (line 41)
        own_type_rule_file_6263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 38), 'own_type_rule_file')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 15), tuple_6261, own_type_rule_file_6263)
        
        # Assigning a type to the variable 'stypy_return_type' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'stypy_return_type', tuple_6261)
        
        # ################# End of '__rule_files(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__rule_files' in the type store
        # Getting the type of 'stypy_return_type' (line 25)
        stypy_return_type_6264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6264)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__rule_files'
        return stypy_return_type_6264


    @staticmethod
    @norecursion
    def __dependent_type_in_rule_params(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__dependent_type_in_rule_params'
        module_type_store = module_type_store.open_function_context('__dependent_type_in_rule_params', 43, 4, False)
        
        # Passed parameters checking function
        TypeRuleCallHandler.__dependent_type_in_rule_params.__dict__.__setitem__('stypy_localization', localization)
        TypeRuleCallHandler.__dependent_type_in_rule_params.__dict__.__setitem__('stypy_type_of_self', None)
        TypeRuleCallHandler.__dependent_type_in_rule_params.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeRuleCallHandler.__dependent_type_in_rule_params.__dict__.__setitem__('stypy_function_name', '__dependent_type_in_rule_params')
        TypeRuleCallHandler.__dependent_type_in_rule_params.__dict__.__setitem__('stypy_param_names_list', ['params'])
        TypeRuleCallHandler.__dependent_type_in_rule_params.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeRuleCallHandler.__dependent_type_in_rule_params.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeRuleCallHandler.__dependent_type_in_rule_params.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeRuleCallHandler.__dependent_type_in_rule_params.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeRuleCallHandler.__dependent_type_in_rule_params.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeRuleCallHandler.__dependent_type_in_rule_params.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, None, module_type_store, '__dependent_type_in_rule_params', ['params'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__dependent_type_in_rule_params', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__dependent_type_in_rule_params(...)' code ##################

        str_6265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, (-1)), 'str', '\n        Check if a list of params has dependent types: Types that have to be called somewhat in order to obtain the\n        real type they represent.\n        :param params: List of types\n        :return: bool\n        ')
        
        
        # Call to len(...): (line 51)
        # Processing the call arguments (line 51)
        
        # Call to filter(...): (line 51)
        # Processing the call arguments (line 51)

        @norecursion
        def _stypy_temp_lambda_14(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_14'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_14', 51, 26, True)
            # Passed parameters checking function
            _stypy_temp_lambda_14.stypy_localization = localization
            _stypy_temp_lambda_14.stypy_type_of_self = None
            _stypy_temp_lambda_14.stypy_type_store = module_type_store
            _stypy_temp_lambda_14.stypy_function_name = '_stypy_temp_lambda_14'
            _stypy_temp_lambda_14.stypy_param_names_list = ['par']
            _stypy_temp_lambda_14.stypy_varargs_param_name = None
            _stypy_temp_lambda_14.stypy_kwargs_param_name = None
            _stypy_temp_lambda_14.stypy_call_defaults = defaults
            _stypy_temp_lambda_14.stypy_call_varargs = varargs
            _stypy_temp_lambda_14.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_14', ['par'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_14', ['par'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to isinstance(...): (line 51)
            # Processing the call arguments (line 51)
            # Getting the type of 'par' (line 51)
            par_6269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 49), 'par', False)
            # Getting the type of 'DependentType' (line 51)
            DependentType_6270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 54), 'DependentType', False)
            # Processing the call keyword arguments (line 51)
            kwargs_6271 = {}
            # Getting the type of 'isinstance' (line 51)
            isinstance_6268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 38), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 51)
            isinstance_call_result_6272 = invoke(stypy.reporting.localization.Localization(__file__, 51, 38), isinstance_6268, *[par_6269, DependentType_6270], **kwargs_6271)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 51)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 26), 'stypy_return_type', isinstance_call_result_6272)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_14' in the type store
            # Getting the type of 'stypy_return_type' (line 51)
            stypy_return_type_6273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 26), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_6273)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_14'
            return stypy_return_type_6273

        # Assigning a type to the variable '_stypy_temp_lambda_14' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 26), '_stypy_temp_lambda_14', _stypy_temp_lambda_14)
        # Getting the type of '_stypy_temp_lambda_14' (line 51)
        _stypy_temp_lambda_14_6274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 26), '_stypy_temp_lambda_14')
        # Getting the type of 'params' (line 51)
        params_6275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 70), 'params', False)
        # Processing the call keyword arguments (line 51)
        kwargs_6276 = {}
        # Getting the type of 'filter' (line 51)
        filter_6267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 19), 'filter', False)
        # Calling filter(args, kwargs) (line 51)
        filter_call_result_6277 = invoke(stypy.reporting.localization.Localization(__file__, 51, 19), filter_6267, *[_stypy_temp_lambda_14_6274, params_6275], **kwargs_6276)
        
        # Processing the call keyword arguments (line 51)
        kwargs_6278 = {}
        # Getting the type of 'len' (line 51)
        len_6266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 15), 'len', False)
        # Calling len(args, kwargs) (line 51)
        len_call_result_6279 = invoke(stypy.reporting.localization.Localization(__file__, 51, 15), len_6266, *[filter_call_result_6277], **kwargs_6278)
        
        int_6280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 81), 'int')
        # Applying the binary operator '>' (line 51)
        result_gt_6281 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 15), '>', len_call_result_6279, int_6280)
        
        # Assigning a type to the variable 'stypy_return_type' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'stypy_return_type', result_gt_6281)
        
        # ################# End of '__dependent_type_in_rule_params(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__dependent_type_in_rule_params' in the type store
        # Getting the type of 'stypy_return_type' (line 43)
        stypy_return_type_6282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6282)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__dependent_type_in_rule_params'
        return stypy_return_type_6282


    @staticmethod
    @norecursion
    def __has_varargs_in_rule_params(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__has_varargs_in_rule_params'
        module_type_store = module_type_store.open_function_context('__has_varargs_in_rule_params', 53, 4, False)
        
        # Passed parameters checking function
        TypeRuleCallHandler.__has_varargs_in_rule_params.__dict__.__setitem__('stypy_localization', localization)
        TypeRuleCallHandler.__has_varargs_in_rule_params.__dict__.__setitem__('stypy_type_of_self', None)
        TypeRuleCallHandler.__has_varargs_in_rule_params.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeRuleCallHandler.__has_varargs_in_rule_params.__dict__.__setitem__('stypy_function_name', '__has_varargs_in_rule_params')
        TypeRuleCallHandler.__has_varargs_in_rule_params.__dict__.__setitem__('stypy_param_names_list', ['params'])
        TypeRuleCallHandler.__has_varargs_in_rule_params.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeRuleCallHandler.__has_varargs_in_rule_params.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeRuleCallHandler.__has_varargs_in_rule_params.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeRuleCallHandler.__has_varargs_in_rule_params.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeRuleCallHandler.__has_varargs_in_rule_params.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeRuleCallHandler.__has_varargs_in_rule_params.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, None, module_type_store, '__has_varargs_in_rule_params', ['params'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__has_varargs_in_rule_params', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__has_varargs_in_rule_params(...)' code ##################

        str_6283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, (-1)), 'str', '\n        Check if a list of params has variable number of arguments\n        :param params: List of types\n        :return: bool\n        ')
        
        
        # Call to len(...): (line 60)
        # Processing the call arguments (line 60)
        
        # Call to filter(...): (line 60)
        # Processing the call arguments (line 60)

        @norecursion
        def _stypy_temp_lambda_15(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_15'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_15', 60, 26, True)
            # Passed parameters checking function
            _stypy_temp_lambda_15.stypy_localization = localization
            _stypy_temp_lambda_15.stypy_type_of_self = None
            _stypy_temp_lambda_15.stypy_type_store = module_type_store
            _stypy_temp_lambda_15.stypy_function_name = '_stypy_temp_lambda_15'
            _stypy_temp_lambda_15.stypy_param_names_list = ['par']
            _stypy_temp_lambda_15.stypy_varargs_param_name = None
            _stypy_temp_lambda_15.stypy_kwargs_param_name = None
            _stypy_temp_lambda_15.stypy_call_defaults = defaults
            _stypy_temp_lambda_15.stypy_call_varargs = varargs
            _stypy_temp_lambda_15.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_15', ['par'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_15', ['par'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to isinstance(...): (line 60)
            # Processing the call arguments (line 60)
            # Getting the type of 'par' (line 60)
            par_6287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 49), 'par', False)
            # Getting the type of 'VarArgType' (line 60)
            VarArgType_6288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 54), 'VarArgType', False)
            # Processing the call keyword arguments (line 60)
            kwargs_6289 = {}
            # Getting the type of 'isinstance' (line 60)
            isinstance_6286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 38), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 60)
            isinstance_call_result_6290 = invoke(stypy.reporting.localization.Localization(__file__, 60, 38), isinstance_6286, *[par_6287, VarArgType_6288], **kwargs_6289)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 60)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 26), 'stypy_return_type', isinstance_call_result_6290)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_15' in the type store
            # Getting the type of 'stypy_return_type' (line 60)
            stypy_return_type_6291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 26), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_6291)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_15'
            return stypy_return_type_6291

        # Assigning a type to the variable '_stypy_temp_lambda_15' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 26), '_stypy_temp_lambda_15', _stypy_temp_lambda_15)
        # Getting the type of '_stypy_temp_lambda_15' (line 60)
        _stypy_temp_lambda_15_6292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 26), '_stypy_temp_lambda_15')
        # Getting the type of 'params' (line 60)
        params_6293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 67), 'params', False)
        # Processing the call keyword arguments (line 60)
        kwargs_6294 = {}
        # Getting the type of 'filter' (line 60)
        filter_6285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 19), 'filter', False)
        # Calling filter(args, kwargs) (line 60)
        filter_call_result_6295 = invoke(stypy.reporting.localization.Localization(__file__, 60, 19), filter_6285, *[_stypy_temp_lambda_15_6292, params_6293], **kwargs_6294)
        
        # Processing the call keyword arguments (line 60)
        kwargs_6296 = {}
        # Getting the type of 'len' (line 60)
        len_6284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 15), 'len', False)
        # Calling len(args, kwargs) (line 60)
        len_call_result_6297 = invoke(stypy.reporting.localization.Localization(__file__, 60, 15), len_6284, *[filter_call_result_6295], **kwargs_6296)
        
        int_6298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 78), 'int')
        # Applying the binary operator '>' (line 60)
        result_gt_6299 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 15), '>', len_call_result_6297, int_6298)
        
        # Assigning a type to the variable 'stypy_return_type' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'stypy_return_type', result_gt_6299)
        
        # ################# End of '__has_varargs_in_rule_params(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__has_varargs_in_rule_params' in the type store
        # Getting the type of 'stypy_return_type' (line 53)
        stypy_return_type_6300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6300)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__has_varargs_in_rule_params'
        return stypy_return_type_6300


    @staticmethod
    @norecursion
    def __get_arguments(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__get_arguments'
        module_type_store = module_type_store.open_function_context('__get_arguments', 62, 4, False)
        
        # Passed parameters checking function
        TypeRuleCallHandler.__get_arguments.__dict__.__setitem__('stypy_localization', localization)
        TypeRuleCallHandler.__get_arguments.__dict__.__setitem__('stypy_type_of_self', None)
        TypeRuleCallHandler.__get_arguments.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeRuleCallHandler.__get_arguments.__dict__.__setitem__('stypy_function_name', '__get_arguments')
        TypeRuleCallHandler.__get_arguments.__dict__.__setitem__('stypy_param_names_list', ['argument_tuple', 'current_pos', 'rule_arity'])
        TypeRuleCallHandler.__get_arguments.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeRuleCallHandler.__get_arguments.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeRuleCallHandler.__get_arguments.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeRuleCallHandler.__get_arguments.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeRuleCallHandler.__get_arguments.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeRuleCallHandler.__get_arguments.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, None, module_type_store, '__get_arguments', ['argument_tuple', 'current_pos', 'rule_arity'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__get_arguments', localization, ['current_pos', 'rule_arity'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__get_arguments(...)' code ##################

        str_6301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, (-1)), 'str', '\n        Obtain a list composed by the arguments present in argument_tuple, except the one in current_pos limited\n        to rule_arity size. This is used when invoking dependent rules\n        :param argument_tuple:\n        :param current_pos:\n        :param rule_arity:\n        :return:\n        ')
        
        # Getting the type of 'rule_arity' (line 72)
        rule_arity_6302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 11), 'rule_arity')
        int_6303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 25), 'int')
        # Applying the binary operator '==' (line 72)
        result_eq_6304 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 11), '==', rule_arity_6302, int_6303)
        
        # Testing if the type of an if condition is none (line 72)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 72, 8), result_eq_6304):
            pass
        else:
            
            # Testing the type of an if condition (line 72)
            if_condition_6305 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 72, 8), result_eq_6304)
            # Assigning a type to the variable 'if_condition_6305' (line 72)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'if_condition_6305', if_condition_6305)
            # SSA begins for if statement (line 72)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Obtaining an instance of the builtin type 'list' (line 73)
            list_6306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 19), 'list')
            # Adding type elements to the builtin type 'list' instance (line 73)
            
            # Assigning a type to the variable 'stypy_return_type' (line 73)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 12), 'stypy_return_type', list_6306)
            # SSA join for if statement (line 72)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a List to a Name (line 75):
        
        # Assigning a List to a Name (line 75):
        
        # Obtaining an instance of the builtin type 'list' (line 75)
        list_6307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 75)
        
        # Assigning a type to the variable 'temp_list' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'temp_list', list_6307)
        
        
        # Call to range(...): (line 76)
        # Processing the call arguments (line 76)
        
        # Call to len(...): (line 76)
        # Processing the call arguments (line 76)
        # Getting the type of 'argument_tuple' (line 76)
        argument_tuple_6310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 27), 'argument_tuple', False)
        # Processing the call keyword arguments (line 76)
        kwargs_6311 = {}
        # Getting the type of 'len' (line 76)
        len_6309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 23), 'len', False)
        # Calling len(args, kwargs) (line 76)
        len_call_result_6312 = invoke(stypy.reporting.localization.Localization(__file__, 76, 23), len_6309, *[argument_tuple_6310], **kwargs_6311)
        
        # Processing the call keyword arguments (line 76)
        kwargs_6313 = {}
        # Getting the type of 'range' (line 76)
        range_6308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 17), 'range', False)
        # Calling range(args, kwargs) (line 76)
        range_call_result_6314 = invoke(stypy.reporting.localization.Localization(__file__, 76, 17), range_6308, *[len_call_result_6312], **kwargs_6313)
        
        # Assigning a type to the variable 'range_call_result_6314' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'range_call_result_6314', range_call_result_6314)
        # Testing if the for loop is going to be iterated (line 76)
        # Testing the type of a for loop iterable (line 76)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 76, 8), range_call_result_6314)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 76, 8), range_call_result_6314):
            # Getting the type of the for loop variable (line 76)
            for_loop_var_6315 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 76, 8), range_call_result_6314)
            # Assigning a type to the variable 'i' (line 76)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'i', for_loop_var_6315)
            # SSA begins for a for statement (line 76)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Getting the type of 'i' (line 77)
            i_6316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 19), 'i')
            # Getting the type of 'current_pos' (line 77)
            current_pos_6317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 24), 'current_pos')
            # Applying the binary operator '==' (line 77)
            result_eq_6318 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 19), '==', i_6316, current_pos_6317)
            
            # Applying the 'not' unary operator (line 77)
            result_not__6319 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 15), 'not', result_eq_6318)
            
            # Testing if the type of an if condition is none (line 77)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 77, 12), result_not__6319):
                pass
            else:
                
                # Testing the type of an if condition (line 77)
                if_condition_6320 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 77, 12), result_not__6319)
                # Assigning a type to the variable 'if_condition_6320' (line 77)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 12), 'if_condition_6320', if_condition_6320)
                # SSA begins for if statement (line 77)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to append(...): (line 78)
                # Processing the call arguments (line 78)
                
                # Obtaining the type of the subscript
                # Getting the type of 'i' (line 78)
                i_6323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 48), 'i', False)
                # Getting the type of 'argument_tuple' (line 78)
                argument_tuple_6324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 33), 'argument_tuple', False)
                # Obtaining the member '__getitem__' of a type (line 78)
                getitem___6325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 33), argument_tuple_6324, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 78)
                subscript_call_result_6326 = invoke(stypy.reporting.localization.Localization(__file__, 78, 33), getitem___6325, i_6323)
                
                # Processing the call keyword arguments (line 78)
                kwargs_6327 = {}
                # Getting the type of 'temp_list' (line 78)
                temp_list_6321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 16), 'temp_list', False)
                # Obtaining the member 'append' of a type (line 78)
                append_6322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 16), temp_list_6321, 'append')
                # Calling append(args, kwargs) (line 78)
                append_call_result_6328 = invoke(stypy.reporting.localization.Localization(__file__, 78, 16), append_6322, *[subscript_call_result_6326], **kwargs_6327)
                
                # SSA join for if statement (line 77)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Call to tuple(...): (line 80)
        # Processing the call arguments (line 80)
        
        # Obtaining the type of the subscript
        int_6330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 31), 'int')
        # Getting the type of 'rule_arity' (line 80)
        rule_arity_6331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 33), 'rule_arity', False)
        slice_6332 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 80, 21), int_6330, rule_arity_6331, None)
        # Getting the type of 'temp_list' (line 80)
        temp_list_6333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 21), 'temp_list', False)
        # Obtaining the member '__getitem__' of a type (line 80)
        getitem___6334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 21), temp_list_6333, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 80)
        subscript_call_result_6335 = invoke(stypy.reporting.localization.Localization(__file__, 80, 21), getitem___6334, slice_6332)
        
        # Processing the call keyword arguments (line 80)
        kwargs_6336 = {}
        # Getting the type of 'tuple' (line 80)
        tuple_6329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 15), 'tuple', False)
        # Calling tuple(args, kwargs) (line 80)
        tuple_call_result_6337 = invoke(stypy.reporting.localization.Localization(__file__, 80, 15), tuple_6329, *[subscript_call_result_6335], **kwargs_6336)
        
        # Assigning a type to the variable 'stypy_return_type' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'stypy_return_type', tuple_call_result_6337)
        
        # ################# End of '__get_arguments(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__get_arguments' in the type store
        # Getting the type of 'stypy_return_type' (line 62)
        stypy_return_type_6338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6338)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__get_arguments'
        return stypy_return_type_6338


    @norecursion
    def invoke_dependent_rules(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'invoke_dependent_rules'
        module_type_store = module_type_store.open_function_context('invoke_dependent_rules', 82, 4, False)
        # Assigning a type to the variable 'self' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeRuleCallHandler.invoke_dependent_rules.__dict__.__setitem__('stypy_localization', localization)
        TypeRuleCallHandler.invoke_dependent_rules.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeRuleCallHandler.invoke_dependent_rules.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeRuleCallHandler.invoke_dependent_rules.__dict__.__setitem__('stypy_function_name', 'TypeRuleCallHandler.invoke_dependent_rules')
        TypeRuleCallHandler.invoke_dependent_rules.__dict__.__setitem__('stypy_param_names_list', ['localization', 'rule_params', 'arguments'])
        TypeRuleCallHandler.invoke_dependent_rules.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeRuleCallHandler.invoke_dependent_rules.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeRuleCallHandler.invoke_dependent_rules.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeRuleCallHandler.invoke_dependent_rules.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeRuleCallHandler.invoke_dependent_rules.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeRuleCallHandler.invoke_dependent_rules.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeRuleCallHandler.invoke_dependent_rules', ['localization', 'rule_params', 'arguments'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'invoke_dependent_rules', localization, ['localization', 'rule_params', 'arguments'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'invoke_dependent_rules(...)' code ##################

        str_6339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, (-1)), 'str', '\n        As we said, some rules may contain special types called DependentTypes. These types have to be invoked in\n        order to check that the rule matches with the call or other necessary operations. Dependent types may have\n        several forms, and are called with all the arguments that are checked against the type rule except the one\n        that matches de dependent type, limited by the Dependent type declared rule arity. For example a Dependent\n        Type may be defined like this (see type_groups.py for all the Dependent types defined):\n\n        Overloads__eq__ = HasMember("__eq__", DynamicType, 1)\n\n        This means that Overloads__eq__ matches with all the objects that has a method named __eq__ that has no\n        predefined return type and an arity of 1 parameter. On the other hand, a type rule may be defined like this:\n\n        ((Overloads__eq__, AnyType), DynamicType)\n\n        This means that the type rule matches with a call that has a first argument which overloads the method\n        __eq__ and any kind of second arguments. Although __eq__ is a method that should return bool (is the ==\n        operator) this is not compulsory in Python, the __eq__ method may return anything and this anything will be\n        the result of the rule. So we have to call __eq__ with the second argument (all the arguments but the one\n        that matches with the DependentType limited to the declared dependent type arity), capture and return the\n        result. This is basically the functionality of this method.\n\n        Note that invocation to a method means that the type rule call handler (or another one) may be used again\n        against the invoked method (__eq__ in our example).\n\n        :param localization: Caller information\n        :param rule_params: Rule file entry\n        :param arguments: Arguments passed to the call that matches against the rule file.\n        :return:\n        ')
        
        # Assigning a List to a Name (line 112):
        
        # Assigning a List to a Name (line 112):
        
        # Obtaining an instance of the builtin type 'list' (line 112)
        list_6340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 112)
        
        # Assigning a type to the variable 'temp_rule' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'temp_rule', list_6340)
        
        # Assigning a Name to a Name (line 113):
        
        # Assigning a Name to a Name (line 113):
        # Getting the type of 'False' (line 113)
        False_6341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 29), 'False')
        # Assigning a type to the variable 'needs_reevaluation' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'needs_reevaluation', False_6341)
        
        
        # Call to range(...): (line 114)
        # Processing the call arguments (line 114)
        
        # Call to len(...): (line 114)
        # Processing the call arguments (line 114)
        # Getting the type of 'rule_params' (line 114)
        rule_params_6344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 27), 'rule_params', False)
        # Processing the call keyword arguments (line 114)
        kwargs_6345 = {}
        # Getting the type of 'len' (line 114)
        len_6343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 23), 'len', False)
        # Calling len(args, kwargs) (line 114)
        len_call_result_6346 = invoke(stypy.reporting.localization.Localization(__file__, 114, 23), len_6343, *[rule_params_6344], **kwargs_6345)
        
        # Processing the call keyword arguments (line 114)
        kwargs_6347 = {}
        # Getting the type of 'range' (line 114)
        range_6342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 17), 'range', False)
        # Calling range(args, kwargs) (line 114)
        range_call_result_6348 = invoke(stypy.reporting.localization.Localization(__file__, 114, 17), range_6342, *[len_call_result_6346], **kwargs_6347)
        
        # Assigning a type to the variable 'range_call_result_6348' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'range_call_result_6348', range_call_result_6348)
        # Testing if the for loop is going to be iterated (line 114)
        # Testing the type of a for loop iterable (line 114)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 114, 8), range_call_result_6348)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 114, 8), range_call_result_6348):
            # Getting the type of the for loop variable (line 114)
            for_loop_var_6349 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 114, 8), range_call_result_6348)
            # Assigning a type to the variable 'i' (line 114)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'i', for_loop_var_6349)
            # SSA begins for a for statement (line 114)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to isinstance(...): (line 116)
            # Processing the call arguments (line 116)
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 116)
            i_6351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 38), 'i', False)
            # Getting the type of 'rule_params' (line 116)
            rule_params_6352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 26), 'rule_params', False)
            # Obtaining the member '__getitem__' of a type (line 116)
            getitem___6353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 26), rule_params_6352, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 116)
            subscript_call_result_6354 = invoke(stypy.reporting.localization.Localization(__file__, 116, 26), getitem___6353, i_6351)
            
            # Getting the type of 'DependentType' (line 116)
            DependentType_6355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 42), 'DependentType', False)
            # Processing the call keyword arguments (line 116)
            kwargs_6356 = {}
            # Getting the type of 'isinstance' (line 116)
            isinstance_6350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 15), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 116)
            isinstance_call_result_6357 = invoke(stypy.reporting.localization.Localization(__file__, 116, 15), isinstance_6350, *[subscript_call_result_6354, DependentType_6355], **kwargs_6356)
            
            # Testing if the type of an if condition is none (line 116)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 116, 12), isinstance_call_result_6357):
                
                # Call to append(...): (line 153)
                # Processing the call arguments (line 153)
                
                # Obtaining the type of the subscript
                # Getting the type of 'i' (line 153)
                i_6448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 45), 'i', False)
                # Getting the type of 'rule_params' (line 153)
                rule_params_6449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 33), 'rule_params', False)
                # Obtaining the member '__getitem__' of a type (line 153)
                getitem___6450 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 33), rule_params_6449, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 153)
                subscript_call_result_6451 = invoke(stypy.reporting.localization.Localization(__file__, 153, 33), getitem___6450, i_6448)
                
                # Processing the call keyword arguments (line 153)
                kwargs_6452 = {}
                # Getting the type of 'temp_rule' (line 153)
                temp_rule_6446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 16), 'temp_rule', False)
                # Obtaining the member 'append' of a type (line 153)
                append_6447 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 16), temp_rule_6446, 'append')
                # Calling append(args, kwargs) (line 153)
                append_call_result_6453 = invoke(stypy.reporting.localization.Localization(__file__, 153, 16), append_6447, *[subscript_call_result_6451], **kwargs_6452)
                
            else:
                
                # Testing the type of an if condition (line 116)
                if_condition_6358 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 116, 12), isinstance_call_result_6357)
                # Assigning a type to the variable 'if_condition_6358' (line 116)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 12), 'if_condition_6358', if_condition_6358)
                # SSA begins for if statement (line 116)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Tuple (line 118):
                
                # Assigning a Call to a Name:
                
                # Call to (...): (line 118)
                # Processing the call arguments (line 118)
                # Getting the type of 'localization' (line 119)
                localization_6363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 20), 'localization', False)
                
                # Call to __get_arguments(...): (line 119)
                # Processing the call arguments (line 119)
                # Getting the type of 'arguments' (line 119)
                arguments_6366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 56), 'arguments', False)
                # Getting the type of 'i' (line 119)
                i_6367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 67), 'i', False)
                
                # Obtaining the type of the subscript
                # Getting the type of 'i' (line 119)
                i_6368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 82), 'i', False)
                # Getting the type of 'rule_params' (line 119)
                rule_params_6369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 70), 'rule_params', False)
                # Obtaining the member '__getitem__' of a type (line 119)
                getitem___6370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 70), rule_params_6369, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 119)
                subscript_call_result_6371 = invoke(stypy.reporting.localization.Localization(__file__, 119, 70), getitem___6370, i_6368)
                
                # Obtaining the member 'call_arity' of a type (line 119)
                call_arity_6372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 70), subscript_call_result_6371, 'call_arity')
                # Processing the call keyword arguments (line 119)
                kwargs_6373 = {}
                # Getting the type of 'self' (line 119)
                self_6364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 35), 'self', False)
                # Obtaining the member '__get_arguments' of a type (line 119)
                get_arguments_6365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 35), self_6364, '__get_arguments')
                # Calling __get_arguments(args, kwargs) (line 119)
                get_arguments_call_result_6374 = invoke(stypy.reporting.localization.Localization(__file__, 119, 35), get_arguments_6365, *[arguments_6366, i_6367, call_arity_6372], **kwargs_6373)
                
                # Processing the call keyword arguments (line 118)
                kwargs_6375 = {}
                
                # Obtaining the type of the subscript
                # Getting the type of 'i' (line 118)
                i_6359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 66), 'i', False)
                # Getting the type of 'rule_params' (line 118)
                rule_params_6360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 54), 'rule_params', False)
                # Obtaining the member '__getitem__' of a type (line 118)
                getitem___6361 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 54), rule_params_6360, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 118)
                subscript_call_result_6362 = invoke(stypy.reporting.localization.Localization(__file__, 118, 54), getitem___6361, i_6359)
                
                # Calling (args, kwargs) (line 118)
                _call_result_6376 = invoke(stypy.reporting.localization.Localization(__file__, 118, 54), subscript_call_result_6362, *[localization_6363, get_arguments_call_result_6374], **kwargs_6375)
                
                # Assigning a type to the variable 'call_assignment_6172' (line 118)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 16), 'call_assignment_6172', _call_result_6376)
                
                # Assigning a Call to a Name (line 118):
                
                # Call to stypy_get_value_from_tuple(...):
                # Processing the call arguments
                # Getting the type of 'call_assignment_6172' (line 118)
                call_assignment_6172_6377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 16), 'call_assignment_6172', False)
                # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                stypy_get_value_from_tuple_call_result_6378 = stypy_get_value_from_tuple(call_assignment_6172_6377, 2, 0)
                
                # Assigning a type to the variable 'call_assignment_6173' (line 118)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 16), 'call_assignment_6173', stypy_get_value_from_tuple_call_result_6378)
                
                # Assigning a Name to a Name (line 118):
                # Getting the type of 'call_assignment_6173' (line 118)
                call_assignment_6173_6379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 16), 'call_assignment_6173')
                # Assigning a type to the variable 'correct_invokation' (line 118)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 16), 'correct_invokation', call_assignment_6173_6379)
                
                # Assigning a Call to a Name (line 118):
                
                # Call to stypy_get_value_from_tuple(...):
                # Processing the call arguments
                # Getting the type of 'call_assignment_6172' (line 118)
                call_assignment_6172_6380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 16), 'call_assignment_6172', False)
                # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                stypy_get_value_from_tuple_call_result_6381 = stypy_get_value_from_tuple(call_assignment_6172_6380, 2, 1)
                
                # Assigning a type to the variable 'call_assignment_6174' (line 118)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 16), 'call_assignment_6174', stypy_get_value_from_tuple_call_result_6381)
                
                # Assigning a Name to a Name (line 118):
                # Getting the type of 'call_assignment_6174' (line 118)
                call_assignment_6174_6382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 16), 'call_assignment_6174')
                # Assigning a type to the variable 'equivalent_type' (line 118)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 36), 'equivalent_type', call_assignment_6174_6382)
                
                # Getting the type of 'correct_invokation' (line 122)
                correct_invokation_6383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 23), 'correct_invokation')
                # Applying the 'not' unary operator (line 122)
                result_not__6384 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 19), 'not', correct_invokation_6383)
                
                # Testing if the type of an if condition is none (line 122)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 122, 16), result_not__6384):
                    
                    # Type idiom detected: calculating its left and rigth part (line 127)
                    # Getting the type of 'equivalent_type' (line 127)
                    equivalent_type_6391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 20), 'equivalent_type')
                    # Getting the type of 'None' (line 127)
                    None_6392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 46), 'None')
                    
                    (may_be_6393, more_types_in_union_6394) = may_not_be_none(equivalent_type_6391, None_6392)

                    if may_be_6393:

                        if more_types_in_union_6394:
                            # Runtime conditional SSA (line 127)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                        else:
                            module_type_store = module_type_store

                        
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'i' (line 130)
                        i_6395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 39), 'i')
                        # Getting the type of 'rule_params' (line 130)
                        rule_params_6396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 27), 'rule_params')
                        # Obtaining the member '__getitem__' of a type (line 130)
                        getitem___6397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 27), rule_params_6396, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 130)
                        subscript_call_result_6398 = invoke(stypy.reporting.localization.Localization(__file__, 130, 27), getitem___6397, i_6395)
                        
                        # Obtaining the member 'expected_return_type' of a type (line 130)
                        expected_return_type_6399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 27), subscript_call_result_6398, 'expected_return_type')
                        # Getting the type of 'UndefinedType' (line 130)
                        UndefinedType_6400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 66), 'UndefinedType')
                        # Applying the binary operator 'is' (line 130)
                        result_is__6401 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 27), 'is', expected_return_type_6399, UndefinedType_6400)
                        
                        # Testing if the type of an if condition is none (line 130)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 130, 24), result_is__6401):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 130)
                            if_condition_6402 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 130, 24), result_is__6401)
                            # Assigning a type to the variable 'if_condition_6402' (line 130)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 24), 'if_condition_6402', if_condition_6402)
                            # SSA begins for if statement (line 130)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Assigning a Name to a Name (line 131):
                            
                            # Assigning a Name to a Name (line 131):
                            # Getting the type of 'True' (line 131)
                            True_6403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 49), 'True')
                            # Assigning a type to the variable 'needs_reevaluation' (line 131)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 28), 'needs_reevaluation', True_6403)
                            
                            # Call to append(...): (line 132)
                            # Processing the call arguments (line 132)
                            # Getting the type of 'equivalent_type' (line 132)
                            equivalent_type_6406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 45), 'equivalent_type', False)
                            # Processing the call keyword arguments (line 132)
                            kwargs_6407 = {}
                            # Getting the type of 'temp_rule' (line 132)
                            temp_rule_6404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 28), 'temp_rule', False)
                            # Obtaining the member 'append' of a type (line 132)
                            append_6405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 28), temp_rule_6404, 'append')
                            # Calling append(args, kwargs) (line 132)
                            append_call_result_6408 = invoke(stypy.reporting.localization.Localization(__file__, 132, 28), append_6405, *[equivalent_type_6406], **kwargs_6407)
                            
                            # SSA join for if statement (line 130)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'i' (line 135)
                        i_6409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 39), 'i')
                        # Getting the type of 'rule_params' (line 135)
                        rule_params_6410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 27), 'rule_params')
                        # Obtaining the member '__getitem__' of a type (line 135)
                        getitem___6411 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 27), rule_params_6410, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 135)
                        subscript_call_result_6412 = invoke(stypy.reporting.localization.Localization(__file__, 135, 27), getitem___6411, i_6409)
                        
                        # Obtaining the member 'expected_return_type' of a type (line 135)
                        expected_return_type_6413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 27), subscript_call_result_6412, 'expected_return_type')
                        # Getting the type of 'DynamicType' (line 135)
                        DynamicType_6414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 66), 'DynamicType')
                        # Applying the binary operator 'is' (line 135)
                        result_is__6415 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 27), 'is', expected_return_type_6413, DynamicType_6414)
                        
                        # Testing if the type of an if condition is none (line 135)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 135, 24), result_is__6415):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 135)
                            if_condition_6416 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 135, 24), result_is__6415)
                            # Assigning a type to the variable 'if_condition_6416' (line 135)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 24), 'if_condition_6416', if_condition_6416)
                            # SSA begins for if statement (line 135)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Obtaining an instance of the builtin type 'tuple' (line 136)
                            tuple_6417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 35), 'tuple')
                            # Adding type elements to the builtin type 'tuple' instance (line 136)
                            # Adding element type (line 136)
                            # Getting the type of 'True' (line 136)
                            True_6418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 35), 'True')
                            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 35), tuple_6417, True_6418)
                            # Adding element type (line 136)
                            # Getting the type of 'None' (line 136)
                            None_6419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 41), 'None')
                            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 35), tuple_6417, None_6419)
                            # Adding element type (line 136)
                            # Getting the type of 'needs_reevaluation' (line 136)
                            needs_reevaluation_6420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 47), 'needs_reevaluation')
                            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 35), tuple_6417, needs_reevaluation_6420)
                            # Adding element type (line 136)
                            # Getting the type of 'equivalent_type' (line 136)
                            equivalent_type_6421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 67), 'equivalent_type')
                            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 35), tuple_6417, equivalent_type_6421)
                            
                            # Assigning a type to the variable 'stypy_return_type' (line 136)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 28), 'stypy_return_type', tuple_6417)
                            # SSA join for if statement (line 135)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'i' (line 148)
                        i_6422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 39), 'i')
                        # Getting the type of 'rule_params' (line 148)
                        rule_params_6423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 27), 'rule_params')
                        # Obtaining the member '__getitem__' of a type (line 148)
                        getitem___6424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 27), rule_params_6423, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 148)
                        subscript_call_result_6425 = invoke(stypy.reporting.localization.Localization(__file__, 148, 27), getitem___6424, i_6422)
                        
                        # Obtaining the member 'expected_return_type' of a type (line 148)
                        expected_return_type_6426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 27), subscript_call_result_6425, 'expected_return_type')
                        
                        # Call to get_python_type(...): (line 148)
                        # Processing the call keyword arguments (line 148)
                        kwargs_6429 = {}
                        # Getting the type of 'equivalent_type' (line 148)
                        equivalent_type_6427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 70), 'equivalent_type', False)
                        # Obtaining the member 'get_python_type' of a type (line 148)
                        get_python_type_6428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 70), equivalent_type_6427, 'get_python_type')
                        # Calling get_python_type(args, kwargs) (line 148)
                        get_python_type_call_result_6430 = invoke(stypy.reporting.localization.Localization(__file__, 148, 70), get_python_type_6428, *[], **kwargs_6429)
                        
                        # Applying the binary operator 'isnot' (line 148)
                        result_is_not_6431 = python_operator(stypy.reporting.localization.Localization(__file__, 148, 27), 'isnot', expected_return_type_6426, get_python_type_call_result_6430)
                        
                        # Testing if the type of an if condition is none (line 148)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 148, 24), result_is_not_6431):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 148)
                            if_condition_6432 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 148, 24), result_is_not_6431)
                            # Assigning a type to the variable 'if_condition_6432' (line 148)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 24), 'if_condition_6432', if_condition_6432)
                            # SSA begins for if statement (line 148)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Obtaining an instance of the builtin type 'tuple' (line 149)
                            tuple_6433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 35), 'tuple')
                            # Adding type elements to the builtin type 'tuple' instance (line 149)
                            # Adding element type (line 149)
                            # Getting the type of 'False' (line 149)
                            False_6434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 35), 'False')
                            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 35), tuple_6433, False_6434)
                            # Adding element type (line 149)
                            # Getting the type of 'None' (line 149)
                            None_6435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 42), 'None')
                            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 35), tuple_6433, None_6435)
                            # Adding element type (line 149)
                            # Getting the type of 'needs_reevaluation' (line 149)
                            needs_reevaluation_6436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 48), 'needs_reevaluation')
                            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 35), tuple_6433, needs_reevaluation_6436)
                            # Adding element type (line 149)
                            # Getting the type of 'None' (line 149)
                            None_6437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 68), 'None')
                            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 35), tuple_6433, None_6437)
                            
                            # Assigning a type to the variable 'stypy_return_type' (line 149)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 28), 'stypy_return_type', tuple_6433)
                            # SSA join for if statement (line 148)
                            module_type_store = module_type_store.join_ssa_context()
                            


                        if more_types_in_union_6394:
                            # Runtime conditional SSA for else branch (line 127)
                            module_type_store.open_ssa_branch('idiom else')



                    if ((not may_be_6393) or more_types_in_union_6394):
                        
                        # Call to append(...): (line 151)
                        # Processing the call arguments (line 151)
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'i' (line 151)
                        i_6440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 53), 'i', False)
                        # Getting the type of 'rule_params' (line 151)
                        rule_params_6441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 41), 'rule_params', False)
                        # Obtaining the member '__getitem__' of a type (line 151)
                        getitem___6442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 41), rule_params_6441, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 151)
                        subscript_call_result_6443 = invoke(stypy.reporting.localization.Localization(__file__, 151, 41), getitem___6442, i_6440)
                        
                        # Processing the call keyword arguments (line 151)
                        kwargs_6444 = {}
                        # Getting the type of 'temp_rule' (line 151)
                        temp_rule_6438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 24), 'temp_rule', False)
                        # Obtaining the member 'append' of a type (line 151)
                        append_6439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 24), temp_rule_6438, 'append')
                        # Calling append(args, kwargs) (line 151)
                        append_call_result_6445 = invoke(stypy.reporting.localization.Localization(__file__, 151, 24), append_6439, *[subscript_call_result_6443], **kwargs_6444)
                        

                        if (may_be_6393 and more_types_in_union_6394):
                            # SSA join for if statement (line 127)
                            module_type_store = module_type_store.join_ssa_context()


                    
                else:
                    
                    # Testing the type of an if condition (line 122)
                    if_condition_6385 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 122, 16), result_not__6384)
                    # Assigning a type to the variable 'if_condition_6385' (line 122)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 16), 'if_condition_6385', if_condition_6385)
                    # SSA begins for if statement (line 122)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Obtaining an instance of the builtin type 'tuple' (line 124)
                    tuple_6386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 27), 'tuple')
                    # Adding type elements to the builtin type 'tuple' instance (line 124)
                    # Adding element type (line 124)
                    # Getting the type of 'False' (line 124)
                    False_6387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 27), 'False')
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 27), tuple_6386, False_6387)
                    # Adding element type (line 124)
                    # Getting the type of 'None' (line 124)
                    None_6388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 34), 'None')
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 27), tuple_6386, None_6388)
                    # Adding element type (line 124)
                    # Getting the type of 'needs_reevaluation' (line 124)
                    needs_reevaluation_6389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 40), 'needs_reevaluation')
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 27), tuple_6386, needs_reevaluation_6389)
                    # Adding element type (line 124)
                    # Getting the type of 'None' (line 124)
                    None_6390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 60), 'None')
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 27), tuple_6386, None_6390)
                    
                    # Assigning a type to the variable 'stypy_return_type' (line 124)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 20), 'stypy_return_type', tuple_6386)
                    # SSA branch for the else part of an if statement (line 122)
                    module_type_store.open_ssa_branch('else')
                    
                    # Type idiom detected: calculating its left and rigth part (line 127)
                    # Getting the type of 'equivalent_type' (line 127)
                    equivalent_type_6391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 20), 'equivalent_type')
                    # Getting the type of 'None' (line 127)
                    None_6392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 46), 'None')
                    
                    (may_be_6393, more_types_in_union_6394) = may_not_be_none(equivalent_type_6391, None_6392)

                    if may_be_6393:

                        if more_types_in_union_6394:
                            # Runtime conditional SSA (line 127)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                        else:
                            module_type_store = module_type_store

                        
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'i' (line 130)
                        i_6395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 39), 'i')
                        # Getting the type of 'rule_params' (line 130)
                        rule_params_6396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 27), 'rule_params')
                        # Obtaining the member '__getitem__' of a type (line 130)
                        getitem___6397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 27), rule_params_6396, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 130)
                        subscript_call_result_6398 = invoke(stypy.reporting.localization.Localization(__file__, 130, 27), getitem___6397, i_6395)
                        
                        # Obtaining the member 'expected_return_type' of a type (line 130)
                        expected_return_type_6399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 27), subscript_call_result_6398, 'expected_return_type')
                        # Getting the type of 'UndefinedType' (line 130)
                        UndefinedType_6400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 66), 'UndefinedType')
                        # Applying the binary operator 'is' (line 130)
                        result_is__6401 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 27), 'is', expected_return_type_6399, UndefinedType_6400)
                        
                        # Testing if the type of an if condition is none (line 130)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 130, 24), result_is__6401):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 130)
                            if_condition_6402 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 130, 24), result_is__6401)
                            # Assigning a type to the variable 'if_condition_6402' (line 130)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 24), 'if_condition_6402', if_condition_6402)
                            # SSA begins for if statement (line 130)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Assigning a Name to a Name (line 131):
                            
                            # Assigning a Name to a Name (line 131):
                            # Getting the type of 'True' (line 131)
                            True_6403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 49), 'True')
                            # Assigning a type to the variable 'needs_reevaluation' (line 131)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 28), 'needs_reevaluation', True_6403)
                            
                            # Call to append(...): (line 132)
                            # Processing the call arguments (line 132)
                            # Getting the type of 'equivalent_type' (line 132)
                            equivalent_type_6406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 45), 'equivalent_type', False)
                            # Processing the call keyword arguments (line 132)
                            kwargs_6407 = {}
                            # Getting the type of 'temp_rule' (line 132)
                            temp_rule_6404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 28), 'temp_rule', False)
                            # Obtaining the member 'append' of a type (line 132)
                            append_6405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 28), temp_rule_6404, 'append')
                            # Calling append(args, kwargs) (line 132)
                            append_call_result_6408 = invoke(stypy.reporting.localization.Localization(__file__, 132, 28), append_6405, *[equivalent_type_6406], **kwargs_6407)
                            
                            # SSA join for if statement (line 130)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'i' (line 135)
                        i_6409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 39), 'i')
                        # Getting the type of 'rule_params' (line 135)
                        rule_params_6410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 27), 'rule_params')
                        # Obtaining the member '__getitem__' of a type (line 135)
                        getitem___6411 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 27), rule_params_6410, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 135)
                        subscript_call_result_6412 = invoke(stypy.reporting.localization.Localization(__file__, 135, 27), getitem___6411, i_6409)
                        
                        # Obtaining the member 'expected_return_type' of a type (line 135)
                        expected_return_type_6413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 27), subscript_call_result_6412, 'expected_return_type')
                        # Getting the type of 'DynamicType' (line 135)
                        DynamicType_6414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 66), 'DynamicType')
                        # Applying the binary operator 'is' (line 135)
                        result_is__6415 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 27), 'is', expected_return_type_6413, DynamicType_6414)
                        
                        # Testing if the type of an if condition is none (line 135)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 135, 24), result_is__6415):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 135)
                            if_condition_6416 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 135, 24), result_is__6415)
                            # Assigning a type to the variable 'if_condition_6416' (line 135)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 24), 'if_condition_6416', if_condition_6416)
                            # SSA begins for if statement (line 135)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Obtaining an instance of the builtin type 'tuple' (line 136)
                            tuple_6417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 35), 'tuple')
                            # Adding type elements to the builtin type 'tuple' instance (line 136)
                            # Adding element type (line 136)
                            # Getting the type of 'True' (line 136)
                            True_6418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 35), 'True')
                            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 35), tuple_6417, True_6418)
                            # Adding element type (line 136)
                            # Getting the type of 'None' (line 136)
                            None_6419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 41), 'None')
                            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 35), tuple_6417, None_6419)
                            # Adding element type (line 136)
                            # Getting the type of 'needs_reevaluation' (line 136)
                            needs_reevaluation_6420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 47), 'needs_reevaluation')
                            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 35), tuple_6417, needs_reevaluation_6420)
                            # Adding element type (line 136)
                            # Getting the type of 'equivalent_type' (line 136)
                            equivalent_type_6421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 67), 'equivalent_type')
                            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 35), tuple_6417, equivalent_type_6421)
                            
                            # Assigning a type to the variable 'stypy_return_type' (line 136)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 28), 'stypy_return_type', tuple_6417)
                            # SSA join for if statement (line 135)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'i' (line 148)
                        i_6422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 39), 'i')
                        # Getting the type of 'rule_params' (line 148)
                        rule_params_6423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 27), 'rule_params')
                        # Obtaining the member '__getitem__' of a type (line 148)
                        getitem___6424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 27), rule_params_6423, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 148)
                        subscript_call_result_6425 = invoke(stypy.reporting.localization.Localization(__file__, 148, 27), getitem___6424, i_6422)
                        
                        # Obtaining the member 'expected_return_type' of a type (line 148)
                        expected_return_type_6426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 27), subscript_call_result_6425, 'expected_return_type')
                        
                        # Call to get_python_type(...): (line 148)
                        # Processing the call keyword arguments (line 148)
                        kwargs_6429 = {}
                        # Getting the type of 'equivalent_type' (line 148)
                        equivalent_type_6427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 70), 'equivalent_type', False)
                        # Obtaining the member 'get_python_type' of a type (line 148)
                        get_python_type_6428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 70), equivalent_type_6427, 'get_python_type')
                        # Calling get_python_type(args, kwargs) (line 148)
                        get_python_type_call_result_6430 = invoke(stypy.reporting.localization.Localization(__file__, 148, 70), get_python_type_6428, *[], **kwargs_6429)
                        
                        # Applying the binary operator 'isnot' (line 148)
                        result_is_not_6431 = python_operator(stypy.reporting.localization.Localization(__file__, 148, 27), 'isnot', expected_return_type_6426, get_python_type_call_result_6430)
                        
                        # Testing if the type of an if condition is none (line 148)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 148, 24), result_is_not_6431):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 148)
                            if_condition_6432 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 148, 24), result_is_not_6431)
                            # Assigning a type to the variable 'if_condition_6432' (line 148)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 24), 'if_condition_6432', if_condition_6432)
                            # SSA begins for if statement (line 148)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Obtaining an instance of the builtin type 'tuple' (line 149)
                            tuple_6433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 35), 'tuple')
                            # Adding type elements to the builtin type 'tuple' instance (line 149)
                            # Adding element type (line 149)
                            # Getting the type of 'False' (line 149)
                            False_6434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 35), 'False')
                            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 35), tuple_6433, False_6434)
                            # Adding element type (line 149)
                            # Getting the type of 'None' (line 149)
                            None_6435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 42), 'None')
                            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 35), tuple_6433, None_6435)
                            # Adding element type (line 149)
                            # Getting the type of 'needs_reevaluation' (line 149)
                            needs_reevaluation_6436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 48), 'needs_reevaluation')
                            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 35), tuple_6433, needs_reevaluation_6436)
                            # Adding element type (line 149)
                            # Getting the type of 'None' (line 149)
                            None_6437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 68), 'None')
                            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 35), tuple_6433, None_6437)
                            
                            # Assigning a type to the variable 'stypy_return_type' (line 149)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 28), 'stypy_return_type', tuple_6433)
                            # SSA join for if statement (line 148)
                            module_type_store = module_type_store.join_ssa_context()
                            


                        if more_types_in_union_6394:
                            # Runtime conditional SSA for else branch (line 127)
                            module_type_store.open_ssa_branch('idiom else')



                    if ((not may_be_6393) or more_types_in_union_6394):
                        
                        # Call to append(...): (line 151)
                        # Processing the call arguments (line 151)
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'i' (line 151)
                        i_6440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 53), 'i', False)
                        # Getting the type of 'rule_params' (line 151)
                        rule_params_6441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 41), 'rule_params', False)
                        # Obtaining the member '__getitem__' of a type (line 151)
                        getitem___6442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 41), rule_params_6441, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 151)
                        subscript_call_result_6443 = invoke(stypy.reporting.localization.Localization(__file__, 151, 41), getitem___6442, i_6440)
                        
                        # Processing the call keyword arguments (line 151)
                        kwargs_6444 = {}
                        # Getting the type of 'temp_rule' (line 151)
                        temp_rule_6438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 24), 'temp_rule', False)
                        # Obtaining the member 'append' of a type (line 151)
                        append_6439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 24), temp_rule_6438, 'append')
                        # Calling append(args, kwargs) (line 151)
                        append_call_result_6445 = invoke(stypy.reporting.localization.Localization(__file__, 151, 24), append_6439, *[subscript_call_result_6443], **kwargs_6444)
                        

                        if (may_be_6393 and more_types_in_union_6394):
                            # SSA join for if statement (line 127)
                            module_type_store = module_type_store.join_ssa_context()


                    
                    # SSA join for if statement (line 122)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA branch for the else part of an if statement (line 116)
                module_type_store.open_ssa_branch('else')
                
                # Call to append(...): (line 153)
                # Processing the call arguments (line 153)
                
                # Obtaining the type of the subscript
                # Getting the type of 'i' (line 153)
                i_6448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 45), 'i', False)
                # Getting the type of 'rule_params' (line 153)
                rule_params_6449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 33), 'rule_params', False)
                # Obtaining the member '__getitem__' of a type (line 153)
                getitem___6450 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 33), rule_params_6449, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 153)
                subscript_call_result_6451 = invoke(stypy.reporting.localization.Localization(__file__, 153, 33), getitem___6450, i_6448)
                
                # Processing the call keyword arguments (line 153)
                kwargs_6452 = {}
                # Getting the type of 'temp_rule' (line 153)
                temp_rule_6446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 16), 'temp_rule', False)
                # Obtaining the member 'append' of a type (line 153)
                append_6447 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 16), temp_rule_6446, 'append')
                # Calling append(args, kwargs) (line 153)
                append_call_result_6453 = invoke(stypy.reporting.localization.Localization(__file__, 153, 16), append_6447, *[subscript_call_result_6451], **kwargs_6452)
                
                # SSA join for if statement (line 116)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Obtaining an instance of the builtin type 'tuple' (line 154)
        tuple_6454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 154)
        # Adding element type (line 154)
        # Getting the type of 'True' (line 154)
        True_6455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 15), 'True')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 15), tuple_6454, True_6455)
        # Adding element type (line 154)
        
        # Call to tuple(...): (line 154)
        # Processing the call arguments (line 154)
        # Getting the type of 'temp_rule' (line 154)
        temp_rule_6457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 27), 'temp_rule', False)
        # Processing the call keyword arguments (line 154)
        kwargs_6458 = {}
        # Getting the type of 'tuple' (line 154)
        tuple_6456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 21), 'tuple', False)
        # Calling tuple(args, kwargs) (line 154)
        tuple_call_result_6459 = invoke(stypy.reporting.localization.Localization(__file__, 154, 21), tuple_6456, *[temp_rule_6457], **kwargs_6458)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 15), tuple_6454, tuple_call_result_6459)
        # Adding element type (line 154)
        # Getting the type of 'needs_reevaluation' (line 154)
        needs_reevaluation_6460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 39), 'needs_reevaluation')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 15), tuple_6454, needs_reevaluation_6460)
        # Adding element type (line 154)
        # Getting the type of 'None' (line 154)
        None_6461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 59), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 15), tuple_6454, None_6461)
        
        # Assigning a type to the variable 'stypy_return_type' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'stypy_return_type', tuple_6454)
        
        # ################# End of 'invoke_dependent_rules(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'invoke_dependent_rules' in the type store
        # Getting the type of 'stypy_return_type' (line 82)
        stypy_return_type_6462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6462)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'invoke_dependent_rules'
        return stypy_return_type_6462


    @norecursion
    def applies_to(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'applies_to'
        module_type_store = module_type_store.open_function_context('applies_to', 175, 4, False)
        # Assigning a type to the variable 'self' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeRuleCallHandler.applies_to.__dict__.__setitem__('stypy_localization', localization)
        TypeRuleCallHandler.applies_to.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeRuleCallHandler.applies_to.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeRuleCallHandler.applies_to.__dict__.__setitem__('stypy_function_name', 'TypeRuleCallHandler.applies_to')
        TypeRuleCallHandler.applies_to.__dict__.__setitem__('stypy_param_names_list', ['proxy_obj', 'callable_entity'])
        TypeRuleCallHandler.applies_to.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeRuleCallHandler.applies_to.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeRuleCallHandler.applies_to.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeRuleCallHandler.applies_to.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeRuleCallHandler.applies_to.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeRuleCallHandler.applies_to.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeRuleCallHandler.applies_to', ['proxy_obj', 'callable_entity'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'applies_to', localization, ['proxy_obj', 'callable_entity'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'applies_to(...)' code ##################

        str_6463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, (-1)), 'str', '\n        This method determines if this call handler is able to respond to a call to callable_entity. The call handler\n        respond to any callable code that has a rule file associated. This method search the rule file and, if found,\n        loads and caches it for performance reasons. Cache also allows us to not to look for the same file on the\n        hard disk over and over, saving much time. callable_entity rule files have priority over the rule files of\n        their parent entity should both exist.\n\n        :param proxy_obj: TypeInferenceProxy that hold the callable entity\n        :param callable_entity: Callable entity\n        :return: bool\n        ')
        
        # Call to isclass(...): (line 188)
        # Processing the call arguments (line 188)
        # Getting the type of 'callable_entity' (line 188)
        callable_entity_6466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 27), 'callable_entity', False)
        # Processing the call keyword arguments (line 188)
        kwargs_6467 = {}
        # Getting the type of 'inspect' (line 188)
        inspect_6464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 11), 'inspect', False)
        # Obtaining the member 'isclass' of a type (line 188)
        isclass_6465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 11), inspect_6464, 'isclass')
        # Calling isclass(args, kwargs) (line 188)
        isclass_call_result_6468 = invoke(stypy.reporting.localization.Localization(__file__, 188, 11), isclass_6465, *[callable_entity_6466], **kwargs_6467)
        
        # Testing if the type of an if condition is none (line 188)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 188, 8), isclass_call_result_6468):
            
            # Assigning a Attribute to a Name (line 191):
            
            # Assigning a Attribute to a Name (line 191):
            # Getting the type of 'proxy_obj' (line 191)
            proxy_obj_6474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 25), 'proxy_obj')
            # Obtaining the member 'name' of a type (line 191)
            name_6475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 25), proxy_obj_6474, 'name')
            # Assigning a type to the variable 'cache_name' (line 191)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 12), 'cache_name', name_6475)
        else:
            
            # Testing the type of an if condition (line 188)
            if_condition_6469 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 188, 8), isclass_call_result_6468)
            # Assigning a type to the variable 'if_condition_6469' (line 188)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 8), 'if_condition_6469', if_condition_6469)
            # SSA begins for if statement (line 188)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a BinOp to a Name (line 189):
            
            # Assigning a BinOp to a Name (line 189):
            # Getting the type of 'proxy_obj' (line 189)
            proxy_obj_6470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 25), 'proxy_obj')
            # Obtaining the member 'name' of a type (line 189)
            name_6471 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 25), proxy_obj_6470, 'name')
            str_6472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 42), 'str', '.__init__')
            # Applying the binary operator '+' (line 189)
            result_add_6473 = python_operator(stypy.reporting.localization.Localization(__file__, 189, 25), '+', name_6471, str_6472)
            
            # Assigning a type to the variable 'cache_name' (line 189)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 12), 'cache_name', result_add_6473)
            # SSA branch for the else part of an if statement (line 188)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Attribute to a Name (line 191):
            
            # Assigning a Attribute to a Name (line 191):
            # Getting the type of 'proxy_obj' (line 191)
            proxy_obj_6474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 25), 'proxy_obj')
            # Obtaining the member 'name' of a type (line 191)
            name_6475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 25), proxy_obj_6474, 'name')
            # Assigning a type to the variable 'cache_name' (line 191)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 12), 'cache_name', name_6475)
            # SSA join for if statement (line 188)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to get(...): (line 194)
        # Processing the call arguments (line 194)
        # Getting the type of 'cache_name' (line 194)
        cache_name_6479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 48), 'cache_name', False)
        # Getting the type of 'False' (line 194)
        False_6480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 60), 'False', False)
        # Processing the call keyword arguments (line 194)
        kwargs_6481 = {}
        # Getting the type of 'self' (line 194)
        self_6476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 11), 'self', False)
        # Obtaining the member 'unavailable_type_rule_cache' of a type (line 194)
        unavailable_type_rule_cache_6477 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 11), self_6476, 'unavailable_type_rule_cache')
        # Obtaining the member 'get' of a type (line 194)
        get_6478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 11), unavailable_type_rule_cache_6477, 'get')
        # Calling get(args, kwargs) (line 194)
        get_call_result_6482 = invoke(stypy.reporting.localization.Localization(__file__, 194, 11), get_6478, *[cache_name_6479, False_6480], **kwargs_6481)
        
        # Testing if the type of an if condition is none (line 194)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 194, 8), get_call_result_6482):
            pass
        else:
            
            # Testing the type of an if condition (line 194)
            if_condition_6483 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 194, 8), get_call_result_6482)
            # Assigning a type to the variable 'if_condition_6483' (line 194)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 8), 'if_condition_6483', if_condition_6483)
            # SSA begins for if statement (line 194)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'False' (line 195)
            False_6484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 19), 'False')
            # Assigning a type to the variable 'stypy_return_type' (line 195)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 12), 'stypy_return_type', False_6484)
            # SSA join for if statement (line 194)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to get(...): (line 198)
        # Processing the call arguments (line 198)
        # Getting the type of 'cache_name' (line 198)
        cache_name_6488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 36), 'cache_name', False)
        # Getting the type of 'False' (line 198)
        False_6489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 48), 'False', False)
        # Processing the call keyword arguments (line 198)
        kwargs_6490 = {}
        # Getting the type of 'self' (line 198)
        self_6485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 11), 'self', False)
        # Obtaining the member 'type_rule_cache' of a type (line 198)
        type_rule_cache_6486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 11), self_6485, 'type_rule_cache')
        # Obtaining the member 'get' of a type (line 198)
        get_6487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 11), type_rule_cache_6486, 'get')
        # Calling get(args, kwargs) (line 198)
        get_call_result_6491 = invoke(stypy.reporting.localization.Localization(__file__, 198, 11), get_6487, *[cache_name_6488, False_6489], **kwargs_6490)
        
        # Testing if the type of an if condition is none (line 198)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 198, 8), get_call_result_6491):
            pass
        else:
            
            # Testing the type of an if condition (line 198)
            if_condition_6492 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 198, 8), get_call_result_6491)
            # Assigning a type to the variable 'if_condition_6492' (line 198)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'if_condition_6492', if_condition_6492)
            # SSA begins for if statement (line 198)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'True' (line 199)
            True_6493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 19), 'True')
            # Assigning a type to the variable 'stypy_return_type' (line 199)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 12), 'stypy_return_type', True_6493)
            # SSA join for if statement (line 198)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 'proxy_obj' (line 202)
        proxy_obj_6494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 11), 'proxy_obj')
        # Obtaining the member 'parent_proxy' of a type (line 202)
        parent_proxy_6495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 11), proxy_obj_6494, 'parent_proxy')
        # Getting the type of 'None' (line 202)
        None_6496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 41), 'None')
        # Applying the binary operator 'isnot' (line 202)
        result_is_not_6497 = python_operator(stypy.reporting.localization.Localization(__file__, 202, 11), 'isnot', parent_proxy_6495, None_6496)
        
        # Testing if the type of an if condition is none (line 202)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 202, 8), result_is_not_6497):
            pass
        else:
            
            # Testing the type of an if condition (line 202)
            if_condition_6498 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 202, 8), result_is_not_6497)
            # Assigning a type to the variable 'if_condition_6498' (line 202)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'if_condition_6498', if_condition_6498)
            # SSA begins for if statement (line 202)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to get(...): (line 203)
            # Processing the call arguments (line 203)
            # Getting the type of 'proxy_obj' (line 203)
            proxy_obj_6502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 40), 'proxy_obj', False)
            # Obtaining the member 'parent_proxy' of a type (line 203)
            parent_proxy_6503 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 40), proxy_obj_6502, 'parent_proxy')
            # Obtaining the member 'name' of a type (line 203)
            name_6504 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 40), parent_proxy_6503, 'name')
            # Getting the type of 'False' (line 203)
            False_6505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 69), 'False', False)
            # Processing the call keyword arguments (line 203)
            kwargs_6506 = {}
            # Getting the type of 'self' (line 203)
            self_6499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 15), 'self', False)
            # Obtaining the member 'type_rule_cache' of a type (line 203)
            type_rule_cache_6500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 15), self_6499, 'type_rule_cache')
            # Obtaining the member 'get' of a type (line 203)
            get_6501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 15), type_rule_cache_6500, 'get')
            # Calling get(args, kwargs) (line 203)
            get_call_result_6507 = invoke(stypy.reporting.localization.Localization(__file__, 203, 15), get_6501, *[name_6504, False_6505], **kwargs_6506)
            
            # Testing if the type of an if condition is none (line 203)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 203, 12), get_call_result_6507):
                pass
            else:
                
                # Testing the type of an if condition (line 203)
                if_condition_6508 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 203, 12), get_call_result_6507)
                # Assigning a type to the variable 'if_condition_6508' (line 203)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 12), 'if_condition_6508', if_condition_6508)
                # SSA begins for if statement (line 203)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # Getting the type of 'True' (line 204)
                True_6509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 23), 'True')
                # Assigning a type to the variable 'stypy_return_type' (line 204)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 16), 'stypy_return_type', True_6509)
                # SSA join for if statement (line 203)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 202)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Evaluating a boolean operation
        
        # Call to ismethod(...): (line 218)
        # Processing the call arguments (line 218)
        # Getting the type of 'callable_entity' (line 218)
        callable_entity_6512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 28), 'callable_entity', False)
        # Processing the call keyword arguments (line 218)
        kwargs_6513 = {}
        # Getting the type of 'inspect' (line 218)
        inspect_6510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 11), 'inspect', False)
        # Obtaining the member 'ismethod' of a type (line 218)
        ismethod_6511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 11), inspect_6510, 'ismethod')
        # Calling ismethod(args, kwargs) (line 218)
        ismethod_call_result_6514 = invoke(stypy.reporting.localization.Localization(__file__, 218, 11), ismethod_6511, *[callable_entity_6512], **kwargs_6513)
        
        
        # Call to ismethoddescriptor(...): (line 218)
        # Processing the call arguments (line 218)
        # Getting the type of 'callable_entity' (line 218)
        callable_entity_6517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 75), 'callable_entity', False)
        # Processing the call keyword arguments (line 218)
        kwargs_6518 = {}
        # Getting the type of 'inspect' (line 218)
        inspect_6515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 48), 'inspect', False)
        # Obtaining the member 'ismethoddescriptor' of a type (line 218)
        ismethoddescriptor_6516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 48), inspect_6515, 'ismethoddescriptor')
        # Calling ismethoddescriptor(args, kwargs) (line 218)
        ismethoddescriptor_call_result_6519 = invoke(stypy.reporting.localization.Localization(__file__, 218, 48), ismethoddescriptor_6516, *[callable_entity_6517], **kwargs_6518)
        
        # Applying the binary operator 'or' (line 218)
        result_or_keyword_6520 = python_operator(stypy.reporting.localization.Localization(__file__, 218, 11), 'or', ismethod_call_result_6514, ismethoddescriptor_call_result_6519)
        
        # Evaluating a boolean operation
        
        # Call to isbuiltin(...): (line 219)
        # Processing the call arguments (line 219)
        # Getting the type of 'callable_entity' (line 219)
        callable_entity_6523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 38), 'callable_entity', False)
        # Processing the call keyword arguments (line 219)
        kwargs_6524 = {}
        # Getting the type of 'inspect' (line 219)
        inspect_6521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 20), 'inspect', False)
        # Obtaining the member 'isbuiltin' of a type (line 219)
        isbuiltin_6522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 20), inspect_6521, 'isbuiltin')
        # Calling isbuiltin(args, kwargs) (line 219)
        isbuiltin_call_result_6525 = invoke(stypy.reporting.localization.Localization(__file__, 219, 20), isbuiltin_6522, *[callable_entity_6523], **kwargs_6524)
        
        
        # Call to isclass(...): (line 220)
        # Processing the call arguments (line 220)
        
        # Call to get_python_entity(...): (line 220)
        # Processing the call keyword arguments (line 220)
        kwargs_6531 = {}
        # Getting the type of 'proxy_obj' (line 220)
        proxy_obj_6528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 37), 'proxy_obj', False)
        # Obtaining the member 'parent_proxy' of a type (line 220)
        parent_proxy_6529 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 37), proxy_obj_6528, 'parent_proxy')
        # Obtaining the member 'get_python_entity' of a type (line 220)
        get_python_entity_6530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 37), parent_proxy_6529, 'get_python_entity')
        # Calling get_python_entity(args, kwargs) (line 220)
        get_python_entity_call_result_6532 = invoke(stypy.reporting.localization.Localization(__file__, 220, 37), get_python_entity_6530, *[], **kwargs_6531)
        
        # Processing the call keyword arguments (line 220)
        kwargs_6533 = {}
        # Getting the type of 'inspect' (line 220)
        inspect_6526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 21), 'inspect', False)
        # Obtaining the member 'isclass' of a type (line 220)
        isclass_6527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 21), inspect_6526, 'isclass')
        # Calling isclass(args, kwargs) (line 220)
        isclass_call_result_6534 = invoke(stypy.reporting.localization.Localization(__file__, 220, 21), isclass_6527, *[get_python_entity_call_result_6532], **kwargs_6533)
        
        # Applying the binary operator 'and' (line 219)
        result_and_keyword_6535 = python_operator(stypy.reporting.localization.Localization(__file__, 219, 20), 'and', isbuiltin_call_result_6525, isclass_call_result_6534)
        
        # Applying the binary operator 'or' (line 218)
        result_or_keyword_6536 = python_operator(stypy.reporting.localization.Localization(__file__, 218, 11), 'or', result_or_keyword_6520, result_and_keyword_6535)
        
        # Testing if the type of an if condition is none (line 218)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 218, 8), result_or_keyword_6536):
            
            # Assigning a Call to a Tuple (line 235):
            
            # Assigning a Call to a Name:
            
            # Call to __rule_files(...): (line 235)
            # Processing the call arguments (line 235)
            # Getting the type of 'proxy_obj' (line 235)
            proxy_obj_6599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 74), 'proxy_obj', False)
            # Obtaining the member 'parent_proxy' of a type (line 235)
            parent_proxy_6600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 74), proxy_obj_6599, 'parent_proxy')
            # Obtaining the member 'name' of a type (line 235)
            name_6601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 74), parent_proxy_6600, 'name')
            # Getting the type of 'proxy_obj' (line 235)
            proxy_obj_6602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 103), 'proxy_obj', False)
            # Obtaining the member 'name' of a type (line 235)
            name_6603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 103), proxy_obj_6602, 'name')
            # Processing the call keyword arguments (line 235)
            kwargs_6604 = {}
            # Getting the type of 'self' (line 235)
            self_6597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 56), 'self', False)
            # Obtaining the member '__rule_files' of a type (line 235)
            rule_files_6598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 56), self_6597, '__rule_files')
            # Calling __rule_files(args, kwargs) (line 235)
            rule_files_call_result_6605 = invoke(stypy.reporting.localization.Localization(__file__, 235, 56), rule_files_6598, *[name_6601, name_6603], **kwargs_6604)
            
            # Assigning a type to the variable 'call_assignment_6184' (line 235)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 12), 'call_assignment_6184', rule_files_call_result_6605)
            
            # Assigning a Call to a Name (line 235):
            
            # Call to stypy_get_value_from_tuple(...):
            # Processing the call arguments
            # Getting the type of 'call_assignment_6184' (line 235)
            call_assignment_6184_6606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 12), 'call_assignment_6184', False)
            # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
            stypy_get_value_from_tuple_call_result_6607 = stypy_get_value_from_tuple(call_assignment_6184_6606, 2, 0)
            
            # Assigning a type to the variable 'call_assignment_6185' (line 235)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 12), 'call_assignment_6185', stypy_get_value_from_tuple_call_result_6607)
            
            # Assigning a Name to a Name (line 235):
            # Getting the type of 'call_assignment_6185' (line 235)
            call_assignment_6185_6608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 12), 'call_assignment_6185')
            # Assigning a type to the variable 'parent_type_rule_file' (line 235)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 12), 'parent_type_rule_file', call_assignment_6185_6608)
            
            # Assigning a Call to a Name (line 235):
            
            # Call to stypy_get_value_from_tuple(...):
            # Processing the call arguments
            # Getting the type of 'call_assignment_6184' (line 235)
            call_assignment_6184_6609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 12), 'call_assignment_6184', False)
            # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
            stypy_get_value_from_tuple_call_result_6610 = stypy_get_value_from_tuple(call_assignment_6184_6609, 2, 1)
            
            # Assigning a type to the variable 'call_assignment_6186' (line 235)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 12), 'call_assignment_6186', stypy_get_value_from_tuple_call_result_6610)
            
            # Assigning a Name to a Name (line 235):
            # Getting the type of 'call_assignment_6186' (line 235)
            call_assignment_6186_6611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 12), 'call_assignment_6186')
            # Assigning a type to the variable 'own_type_rule_file' (line 235)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 35), 'own_type_rule_file', call_assignment_6186_6611)
        else:
            
            # Testing the type of an if condition (line 218)
            if_condition_6537 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 218, 8), result_or_keyword_6536)
            # Assigning a type to the variable 'if_condition_6537' (line 218)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'if_condition_6537', if_condition_6537)
            # SSA begins for if statement (line 218)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            
            # SSA begins for try-except statement (line 221)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
            
            # Assigning a Call to a Tuple (line 222):
            
            # Assigning a Call to a Name:
            
            # Call to __rule_files(...): (line 222)
            # Processing the call arguments (line 222)
            # Getting the type of 'callable_entity' (line 222)
            callable_entity_6540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 78), 'callable_entity', False)
            # Obtaining the member '__objclass__' of a type (line 222)
            objclass___6541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 78), callable_entity_6540, '__objclass__')
            # Obtaining the member '__module__' of a type (line 222)
            module___6542 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 78), objclass___6541, '__module__')
            # Getting the type of 'callable_entity' (line 223)
            callable_entity_6543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 78), 'callable_entity', False)
            # Obtaining the member '__objclass__' of a type (line 223)
            objclass___6544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 78), callable_entity_6543, '__objclass__')
            # Obtaining the member '__name__' of a type (line 223)
            name___6545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 78), objclass___6544, '__name__')
            # Processing the call keyword arguments (line 222)
            kwargs_6546 = {}
            # Getting the type of 'self' (line 222)
            self_6538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 60), 'self', False)
            # Obtaining the member '__rule_files' of a type (line 222)
            rule_files_6539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 60), self_6538, '__rule_files')
            # Calling __rule_files(args, kwargs) (line 222)
            rule_files_call_result_6547 = invoke(stypy.reporting.localization.Localization(__file__, 222, 60), rule_files_6539, *[module___6542, name___6545], **kwargs_6546)
            
            # Assigning a type to the variable 'call_assignment_6175' (line 222)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 16), 'call_assignment_6175', rule_files_call_result_6547)
            
            # Assigning a Call to a Name (line 222):
            
            # Call to stypy_get_value_from_tuple(...):
            # Processing the call arguments
            # Getting the type of 'call_assignment_6175' (line 222)
            call_assignment_6175_6548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 16), 'call_assignment_6175', False)
            # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
            stypy_get_value_from_tuple_call_result_6549 = stypy_get_value_from_tuple(call_assignment_6175_6548, 2, 0)
            
            # Assigning a type to the variable 'call_assignment_6176' (line 222)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 16), 'call_assignment_6176', stypy_get_value_from_tuple_call_result_6549)
            
            # Assigning a Name to a Name (line 222):
            # Getting the type of 'call_assignment_6176' (line 222)
            call_assignment_6176_6550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 16), 'call_assignment_6176')
            # Assigning a type to the variable 'parent_type_rule_file' (line 222)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 16), 'parent_type_rule_file', call_assignment_6176_6550)
            
            # Assigning a Call to a Name (line 222):
            
            # Call to stypy_get_value_from_tuple(...):
            # Processing the call arguments
            # Getting the type of 'call_assignment_6175' (line 222)
            call_assignment_6175_6551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 16), 'call_assignment_6175', False)
            # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
            stypy_get_value_from_tuple_call_result_6552 = stypy_get_value_from_tuple(call_assignment_6175_6551, 2, 1)
            
            # Assigning a type to the variable 'call_assignment_6177' (line 222)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 16), 'call_assignment_6177', stypy_get_value_from_tuple_call_result_6552)
            
            # Assigning a Name to a Name (line 222):
            # Getting the type of 'call_assignment_6177' (line 222)
            call_assignment_6177_6553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 16), 'call_assignment_6177')
            # Assigning a type to the variable 'own_type_rule_file' (line 222)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 39), 'own_type_rule_file', call_assignment_6177_6553)
            # SSA branch for the except part of a try statement (line 221)
            # SSA branch for the except '<any exception>' branch of a try statement (line 221)
            module_type_store.open_ssa_branch('except')
            
            # Call to ismodule(...): (line 226)
            # Processing the call arguments (line 226)
            
            # Call to get_python_entity(...): (line 226)
            # Processing the call keyword arguments (line 226)
            kwargs_6559 = {}
            # Getting the type of 'proxy_obj' (line 226)
            proxy_obj_6556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 36), 'proxy_obj', False)
            # Obtaining the member 'parent_proxy' of a type (line 226)
            parent_proxy_6557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 36), proxy_obj_6556, 'parent_proxy')
            # Obtaining the member 'get_python_entity' of a type (line 226)
            get_python_entity_6558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 36), parent_proxy_6557, 'get_python_entity')
            # Calling get_python_entity(args, kwargs) (line 226)
            get_python_entity_call_result_6560 = invoke(stypy.reporting.localization.Localization(__file__, 226, 36), get_python_entity_6558, *[], **kwargs_6559)
            
            # Processing the call keyword arguments (line 226)
            kwargs_6561 = {}
            # Getting the type of 'inspect' (line 226)
            inspect_6554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 19), 'inspect', False)
            # Obtaining the member 'ismodule' of a type (line 226)
            ismodule_6555 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 19), inspect_6554, 'ismodule')
            # Calling ismodule(args, kwargs) (line 226)
            ismodule_call_result_6562 = invoke(stypy.reporting.localization.Localization(__file__, 226, 19), ismodule_6555, *[get_python_entity_call_result_6560], **kwargs_6561)
            
            # Testing if the type of an if condition is none (line 226)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 226, 16), ismodule_call_result_6562):
                
                # Assigning a Call to a Tuple (line 231):
                
                # Assigning a Call to a Name:
                
                # Call to __rule_files(...): (line 231)
                # Processing the call arguments (line 231)
                # Getting the type of 'proxy_obj' (line 232)
                proxy_obj_6582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 24), 'proxy_obj', False)
                # Obtaining the member 'parent_proxy' of a type (line 232)
                parent_proxy_6583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 24), proxy_obj_6582, 'parent_proxy')
                # Obtaining the member 'parent_proxy' of a type (line 232)
                parent_proxy_6584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 24), parent_proxy_6583, 'parent_proxy')
                # Obtaining the member 'name' of a type (line 232)
                name_6585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 24), parent_proxy_6584, 'name')
                # Getting the type of 'proxy_obj' (line 233)
                proxy_obj_6586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 24), 'proxy_obj', False)
                # Obtaining the member 'parent_proxy' of a type (line 233)
                parent_proxy_6587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 24), proxy_obj_6586, 'parent_proxy')
                # Obtaining the member 'name' of a type (line 233)
                name_6588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 24), parent_proxy_6587, 'name')
                # Processing the call keyword arguments (line 231)
                kwargs_6589 = {}
                # Getting the type of 'self' (line 231)
                self_6580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 64), 'self', False)
                # Obtaining the member '__rule_files' of a type (line 231)
                rule_files_6581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 64), self_6580, '__rule_files')
                # Calling __rule_files(args, kwargs) (line 231)
                rule_files_call_result_6590 = invoke(stypy.reporting.localization.Localization(__file__, 231, 64), rule_files_6581, *[name_6585, name_6588], **kwargs_6589)
                
                # Assigning a type to the variable 'call_assignment_6181' (line 231)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 20), 'call_assignment_6181', rule_files_call_result_6590)
                
                # Assigning a Call to a Name (line 231):
                
                # Call to stypy_get_value_from_tuple(...):
                # Processing the call arguments
                # Getting the type of 'call_assignment_6181' (line 231)
                call_assignment_6181_6591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 20), 'call_assignment_6181', False)
                # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                stypy_get_value_from_tuple_call_result_6592 = stypy_get_value_from_tuple(call_assignment_6181_6591, 2, 0)
                
                # Assigning a type to the variable 'call_assignment_6182' (line 231)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 20), 'call_assignment_6182', stypy_get_value_from_tuple_call_result_6592)
                
                # Assigning a Name to a Name (line 231):
                # Getting the type of 'call_assignment_6182' (line 231)
                call_assignment_6182_6593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 20), 'call_assignment_6182')
                # Assigning a type to the variable 'parent_type_rule_file' (line 231)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 20), 'parent_type_rule_file', call_assignment_6182_6593)
                
                # Assigning a Call to a Name (line 231):
                
                # Call to stypy_get_value_from_tuple(...):
                # Processing the call arguments
                # Getting the type of 'call_assignment_6181' (line 231)
                call_assignment_6181_6594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 20), 'call_assignment_6181', False)
                # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                stypy_get_value_from_tuple_call_result_6595 = stypy_get_value_from_tuple(call_assignment_6181_6594, 2, 1)
                
                # Assigning a type to the variable 'call_assignment_6183' (line 231)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 20), 'call_assignment_6183', stypy_get_value_from_tuple_call_result_6595)
                
                # Assigning a Name to a Name (line 231):
                # Getting the type of 'call_assignment_6183' (line 231)
                call_assignment_6183_6596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 20), 'call_assignment_6183')
                # Assigning a type to the variable 'own_type_rule_file' (line 231)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 43), 'own_type_rule_file', call_assignment_6183_6596)
            else:
                
                # Testing the type of an if condition (line 226)
                if_condition_6563 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 226, 16), ismodule_call_result_6562)
                # Assigning a type to the variable 'if_condition_6563' (line 226)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 16), 'if_condition_6563', if_condition_6563)
                # SSA begins for if statement (line 226)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Tuple (line 227):
                
                # Assigning a Call to a Name:
                
                # Call to __rule_files(...): (line 227)
                # Processing the call arguments (line 227)
                # Getting the type of 'proxy_obj' (line 228)
                proxy_obj_6566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 24), 'proxy_obj', False)
                # Obtaining the member 'parent_proxy' of a type (line 228)
                parent_proxy_6567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 24), proxy_obj_6566, 'parent_proxy')
                # Obtaining the member 'name' of a type (line 228)
                name_6568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 24), parent_proxy_6567, 'name')
                # Getting the type of 'proxy_obj' (line 229)
                proxy_obj_6569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 24), 'proxy_obj', False)
                # Obtaining the member 'parent_proxy' of a type (line 229)
                parent_proxy_6570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 24), proxy_obj_6569, 'parent_proxy')
                # Obtaining the member 'name' of a type (line 229)
                name_6571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 24), parent_proxy_6570, 'name')
                # Processing the call keyword arguments (line 227)
                kwargs_6572 = {}
                # Getting the type of 'self' (line 227)
                self_6564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 64), 'self', False)
                # Obtaining the member '__rule_files' of a type (line 227)
                rule_files_6565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 64), self_6564, '__rule_files')
                # Calling __rule_files(args, kwargs) (line 227)
                rule_files_call_result_6573 = invoke(stypy.reporting.localization.Localization(__file__, 227, 64), rule_files_6565, *[name_6568, name_6571], **kwargs_6572)
                
                # Assigning a type to the variable 'call_assignment_6178' (line 227)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 20), 'call_assignment_6178', rule_files_call_result_6573)
                
                # Assigning a Call to a Name (line 227):
                
                # Call to stypy_get_value_from_tuple(...):
                # Processing the call arguments
                # Getting the type of 'call_assignment_6178' (line 227)
                call_assignment_6178_6574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 20), 'call_assignment_6178', False)
                # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                stypy_get_value_from_tuple_call_result_6575 = stypy_get_value_from_tuple(call_assignment_6178_6574, 2, 0)
                
                # Assigning a type to the variable 'call_assignment_6179' (line 227)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 20), 'call_assignment_6179', stypy_get_value_from_tuple_call_result_6575)
                
                # Assigning a Name to a Name (line 227):
                # Getting the type of 'call_assignment_6179' (line 227)
                call_assignment_6179_6576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 20), 'call_assignment_6179')
                # Assigning a type to the variable 'parent_type_rule_file' (line 227)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 20), 'parent_type_rule_file', call_assignment_6179_6576)
                
                # Assigning a Call to a Name (line 227):
                
                # Call to stypy_get_value_from_tuple(...):
                # Processing the call arguments
                # Getting the type of 'call_assignment_6178' (line 227)
                call_assignment_6178_6577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 20), 'call_assignment_6178', False)
                # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                stypy_get_value_from_tuple_call_result_6578 = stypy_get_value_from_tuple(call_assignment_6178_6577, 2, 1)
                
                # Assigning a type to the variable 'call_assignment_6180' (line 227)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 20), 'call_assignment_6180', stypy_get_value_from_tuple_call_result_6578)
                
                # Assigning a Name to a Name (line 227):
                # Getting the type of 'call_assignment_6180' (line 227)
                call_assignment_6180_6579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 20), 'call_assignment_6180')
                # Assigning a type to the variable 'own_type_rule_file' (line 227)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 43), 'own_type_rule_file', call_assignment_6180_6579)
                # SSA branch for the else part of an if statement (line 226)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Call to a Tuple (line 231):
                
                # Assigning a Call to a Name:
                
                # Call to __rule_files(...): (line 231)
                # Processing the call arguments (line 231)
                # Getting the type of 'proxy_obj' (line 232)
                proxy_obj_6582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 24), 'proxy_obj', False)
                # Obtaining the member 'parent_proxy' of a type (line 232)
                parent_proxy_6583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 24), proxy_obj_6582, 'parent_proxy')
                # Obtaining the member 'parent_proxy' of a type (line 232)
                parent_proxy_6584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 24), parent_proxy_6583, 'parent_proxy')
                # Obtaining the member 'name' of a type (line 232)
                name_6585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 24), parent_proxy_6584, 'name')
                # Getting the type of 'proxy_obj' (line 233)
                proxy_obj_6586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 24), 'proxy_obj', False)
                # Obtaining the member 'parent_proxy' of a type (line 233)
                parent_proxy_6587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 24), proxy_obj_6586, 'parent_proxy')
                # Obtaining the member 'name' of a type (line 233)
                name_6588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 24), parent_proxy_6587, 'name')
                # Processing the call keyword arguments (line 231)
                kwargs_6589 = {}
                # Getting the type of 'self' (line 231)
                self_6580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 64), 'self', False)
                # Obtaining the member '__rule_files' of a type (line 231)
                rule_files_6581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 64), self_6580, '__rule_files')
                # Calling __rule_files(args, kwargs) (line 231)
                rule_files_call_result_6590 = invoke(stypy.reporting.localization.Localization(__file__, 231, 64), rule_files_6581, *[name_6585, name_6588], **kwargs_6589)
                
                # Assigning a type to the variable 'call_assignment_6181' (line 231)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 20), 'call_assignment_6181', rule_files_call_result_6590)
                
                # Assigning a Call to a Name (line 231):
                
                # Call to stypy_get_value_from_tuple(...):
                # Processing the call arguments
                # Getting the type of 'call_assignment_6181' (line 231)
                call_assignment_6181_6591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 20), 'call_assignment_6181', False)
                # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                stypy_get_value_from_tuple_call_result_6592 = stypy_get_value_from_tuple(call_assignment_6181_6591, 2, 0)
                
                # Assigning a type to the variable 'call_assignment_6182' (line 231)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 20), 'call_assignment_6182', stypy_get_value_from_tuple_call_result_6592)
                
                # Assigning a Name to a Name (line 231):
                # Getting the type of 'call_assignment_6182' (line 231)
                call_assignment_6182_6593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 20), 'call_assignment_6182')
                # Assigning a type to the variable 'parent_type_rule_file' (line 231)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 20), 'parent_type_rule_file', call_assignment_6182_6593)
                
                # Assigning a Call to a Name (line 231):
                
                # Call to stypy_get_value_from_tuple(...):
                # Processing the call arguments
                # Getting the type of 'call_assignment_6181' (line 231)
                call_assignment_6181_6594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 20), 'call_assignment_6181', False)
                # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                stypy_get_value_from_tuple_call_result_6595 = stypy_get_value_from_tuple(call_assignment_6181_6594, 2, 1)
                
                # Assigning a type to the variable 'call_assignment_6183' (line 231)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 20), 'call_assignment_6183', stypy_get_value_from_tuple_call_result_6595)
                
                # Assigning a Name to a Name (line 231):
                # Getting the type of 'call_assignment_6183' (line 231)
                call_assignment_6183_6596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 20), 'call_assignment_6183')
                # Assigning a type to the variable 'own_type_rule_file' (line 231)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 43), 'own_type_rule_file', call_assignment_6183_6596)
                # SSA join for if statement (line 226)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for try-except statement (line 221)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA branch for the else part of an if statement (line 218)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Call to a Tuple (line 235):
            
            # Assigning a Call to a Name:
            
            # Call to __rule_files(...): (line 235)
            # Processing the call arguments (line 235)
            # Getting the type of 'proxy_obj' (line 235)
            proxy_obj_6599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 74), 'proxy_obj', False)
            # Obtaining the member 'parent_proxy' of a type (line 235)
            parent_proxy_6600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 74), proxy_obj_6599, 'parent_proxy')
            # Obtaining the member 'name' of a type (line 235)
            name_6601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 74), parent_proxy_6600, 'name')
            # Getting the type of 'proxy_obj' (line 235)
            proxy_obj_6602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 103), 'proxy_obj', False)
            # Obtaining the member 'name' of a type (line 235)
            name_6603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 103), proxy_obj_6602, 'name')
            # Processing the call keyword arguments (line 235)
            kwargs_6604 = {}
            # Getting the type of 'self' (line 235)
            self_6597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 56), 'self', False)
            # Obtaining the member '__rule_files' of a type (line 235)
            rule_files_6598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 56), self_6597, '__rule_files')
            # Calling __rule_files(args, kwargs) (line 235)
            rule_files_call_result_6605 = invoke(stypy.reporting.localization.Localization(__file__, 235, 56), rule_files_6598, *[name_6601, name_6603], **kwargs_6604)
            
            # Assigning a type to the variable 'call_assignment_6184' (line 235)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 12), 'call_assignment_6184', rule_files_call_result_6605)
            
            # Assigning a Call to a Name (line 235):
            
            # Call to stypy_get_value_from_tuple(...):
            # Processing the call arguments
            # Getting the type of 'call_assignment_6184' (line 235)
            call_assignment_6184_6606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 12), 'call_assignment_6184', False)
            # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
            stypy_get_value_from_tuple_call_result_6607 = stypy_get_value_from_tuple(call_assignment_6184_6606, 2, 0)
            
            # Assigning a type to the variable 'call_assignment_6185' (line 235)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 12), 'call_assignment_6185', stypy_get_value_from_tuple_call_result_6607)
            
            # Assigning a Name to a Name (line 235):
            # Getting the type of 'call_assignment_6185' (line 235)
            call_assignment_6185_6608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 12), 'call_assignment_6185')
            # Assigning a type to the variable 'parent_type_rule_file' (line 235)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 12), 'parent_type_rule_file', call_assignment_6185_6608)
            
            # Assigning a Call to a Name (line 235):
            
            # Call to stypy_get_value_from_tuple(...):
            # Processing the call arguments
            # Getting the type of 'call_assignment_6184' (line 235)
            call_assignment_6184_6609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 12), 'call_assignment_6184', False)
            # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
            stypy_get_value_from_tuple_call_result_6610 = stypy_get_value_from_tuple(call_assignment_6184_6609, 2, 1)
            
            # Assigning a type to the variable 'call_assignment_6186' (line 235)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 12), 'call_assignment_6186', stypy_get_value_from_tuple_call_result_6610)
            
            # Assigning a Name to a Name (line 235):
            # Getting the type of 'call_assignment_6186' (line 235)
            call_assignment_6186_6611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 12), 'call_assignment_6186')
            # Assigning a type to the variable 'own_type_rule_file' (line 235)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 35), 'own_type_rule_file', call_assignment_6186_6611)
            # SSA join for if statement (line 218)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Call to a Name (line 238):
        
        # Assigning a Call to a Name (line 238):
        
        # Call to isfile(...): (line 238)
        # Processing the call arguments (line 238)
        # Getting the type of 'parent_type_rule_file' (line 238)
        parent_type_rule_file_6615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 38), 'parent_type_rule_file', False)
        # Processing the call keyword arguments (line 238)
        kwargs_6616 = {}
        # Getting the type of 'os' (line 238)
        os_6612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 23), 'os', False)
        # Obtaining the member 'path' of a type (line 238)
        path_6613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 23), os_6612, 'path')
        # Obtaining the member 'isfile' of a type (line 238)
        isfile_6614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 23), path_6613, 'isfile')
        # Calling isfile(args, kwargs) (line 238)
        isfile_call_result_6617 = invoke(stypy.reporting.localization.Localization(__file__, 238, 23), isfile_6614, *[parent_type_rule_file_6615], **kwargs_6616)
        
        # Assigning a type to the variable 'parent_exist' (line 238)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 8), 'parent_exist', isfile_call_result_6617)
        
        # Assigning a Call to a Name (line 239):
        
        # Assigning a Call to a Name (line 239):
        
        # Call to isfile(...): (line 239)
        # Processing the call arguments (line 239)
        # Getting the type of 'own_type_rule_file' (line 239)
        own_type_rule_file_6621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 35), 'own_type_rule_file', False)
        # Processing the call keyword arguments (line 239)
        kwargs_6622 = {}
        # Getting the type of 'os' (line 239)
        os_6618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 20), 'os', False)
        # Obtaining the member 'path' of a type (line 239)
        path_6619 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 20), os_6618, 'path')
        # Obtaining the member 'isfile' of a type (line 239)
        isfile_6620 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 20), path_6619, 'isfile')
        # Calling isfile(args, kwargs) (line 239)
        isfile_call_result_6623 = invoke(stypy.reporting.localization.Localization(__file__, 239, 20), isfile_6620, *[own_type_rule_file_6621], **kwargs_6622)
        
        # Assigning a type to the variable 'own_exist' (line 239)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 8), 'own_exist', isfile_call_result_6623)
        
        # Assigning a Str to a Name (line 240):
        
        # Assigning a Str to a Name (line 240):
        str_6624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 20), 'str', '')
        # Assigning a type to the variable 'file_path' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'file_path', str_6624)
        # Getting the type of 'parent_exist' (line 242)
        parent_exist_6625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 11), 'parent_exist')
        # Testing if the type of an if condition is none (line 242)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 242, 8), parent_exist_6625):
            pass
        else:
            
            # Testing the type of an if condition (line 242)
            if_condition_6626 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 242, 8), parent_exist_6625)
            # Assigning a type to the variable 'if_condition_6626' (line 242)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 8), 'if_condition_6626', if_condition_6626)
            # SSA begins for if statement (line 242)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Name to a Name (line 243):
            
            # Assigning a Name to a Name (line 243):
            # Getting the type of 'parent_type_rule_file' (line 243)
            parent_type_rule_file_6627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 24), 'parent_type_rule_file')
            # Assigning a type to the variable 'file_path' (line 243)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 12), 'file_path', parent_type_rule_file_6627)
            # SSA join for if statement (line 242)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'own_exist' (line 245)
        own_exist_6628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 11), 'own_exist')
        # Testing if the type of an if condition is none (line 245)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 245, 8), own_exist_6628):
            pass
        else:
            
            # Testing the type of an if condition (line 245)
            if_condition_6629 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 245, 8), own_exist_6628)
            # Assigning a type to the variable 'if_condition_6629' (line 245)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'if_condition_6629', if_condition_6629)
            # SSA begins for if statement (line 245)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Name to a Name (line 246):
            
            # Assigning a Name to a Name (line 246):
            # Getting the type of 'own_type_rule_file' (line 246)
            own_type_rule_file_6630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 24), 'own_type_rule_file')
            # Assigning a type to the variable 'file_path' (line 246)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 12), 'file_path', own_type_rule_file_6630)
            # SSA join for if statement (line 245)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Evaluating a boolean operation
        # Getting the type of 'parent_exist' (line 249)
        parent_exist_6631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 11), 'parent_exist')
        # Getting the type of 'own_exist' (line 249)
        own_exist_6632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 27), 'own_exist')
        # Applying the binary operator 'or' (line 249)
        result_or_keyword_6633 = python_operator(stypy.reporting.localization.Localization(__file__, 249, 11), 'or', parent_exist_6631, own_exist_6632)
        
        # Testing if the type of an if condition is none (line 249)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 249, 8), result_or_keyword_6633):
            pass
        else:
            
            # Testing the type of an if condition (line 249)
            if_condition_6634 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 249, 8), result_or_keyword_6633)
            # Assigning a type to the variable 'if_condition_6634' (line 249)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 8), 'if_condition_6634', if_condition_6634)
            # SSA begins for if statement (line 249)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 250):
            
            # Assigning a Call to a Name (line 250):
            
            # Call to dirname(...): (line 250)
            # Processing the call arguments (line 250)
            # Getting the type of 'file_path' (line 250)
            file_path_6638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 38), 'file_path', False)
            # Processing the call keyword arguments (line 250)
            kwargs_6639 = {}
            # Getting the type of 'os' (line 250)
            os_6635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 22), 'os', False)
            # Obtaining the member 'path' of a type (line 250)
            path_6636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 22), os_6635, 'path')
            # Obtaining the member 'dirname' of a type (line 250)
            dirname_6637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 22), path_6636, 'dirname')
            # Calling dirname(args, kwargs) (line 250)
            dirname_call_result_6640 = invoke(stypy.reporting.localization.Localization(__file__, 250, 22), dirname_6637, *[file_path_6638], **kwargs_6639)
            
            # Assigning a type to the variable 'dirname' (line 250)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 12), 'dirname', dirname_call_result_6640)
            
            # Assigning a Subscript to a Name (line 251):
            
            # Assigning a Subscript to a Name (line 251):
            
            # Obtaining the type of the subscript
            int_6641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 45), 'int')
            int_6642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 47), 'int')
            slice_6643 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 251, 20), int_6641, int_6642, None)
            
            # Obtaining the type of the subscript
            int_6644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 41), 'int')
            
            # Call to split(...): (line 251)
            # Processing the call arguments (line 251)
            str_6647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 36), 'str', '/')
            # Processing the call keyword arguments (line 251)
            kwargs_6648 = {}
            # Getting the type of 'file_path' (line 251)
            file_path_6645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 20), 'file_path', False)
            # Obtaining the member 'split' of a type (line 251)
            split_6646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 20), file_path_6645, 'split')
            # Calling split(args, kwargs) (line 251)
            split_call_result_6649 = invoke(stypy.reporting.localization.Localization(__file__, 251, 20), split_6646, *[str_6647], **kwargs_6648)
            
            # Obtaining the member '__getitem__' of a type (line 251)
            getitem___6650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 20), split_call_result_6649, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 251)
            subscript_call_result_6651 = invoke(stypy.reporting.localization.Localization(__file__, 251, 20), getitem___6650, int_6644)
            
            # Obtaining the member '__getitem__' of a type (line 251)
            getitem___6652 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 20), subscript_call_result_6651, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 251)
            subscript_call_result_6653 = invoke(stypy.reporting.localization.Localization(__file__, 251, 20), getitem___6652, slice_6643)
            
            # Assigning a type to the variable 'file_' (line 251)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 12), 'file_', subscript_call_result_6653)
            
            # Call to append(...): (line 253)
            # Processing the call arguments (line 253)
            # Getting the type of 'dirname' (line 253)
            dirname_6657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 28), 'dirname', False)
            # Processing the call keyword arguments (line 253)
            kwargs_6658 = {}
            # Getting the type of 'sys' (line 253)
            sys_6654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 12), 'sys', False)
            # Obtaining the member 'path' of a type (line 253)
            path_6655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 12), sys_6654, 'path')
            # Obtaining the member 'append' of a type (line 253)
            append_6656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 12), path_6655, 'append')
            # Calling append(args, kwargs) (line 253)
            append_call_result_6659 = invoke(stypy.reporting.localization.Localization(__file__, 253, 12), append_6656, *[dirname_6657], **kwargs_6658)
            
            
            # Assigning a Call to a Name (line 254):
            
            # Assigning a Call to a Name (line 254):
            
            # Call to __import__(...): (line 254)
            # Processing the call arguments (line 254)
            # Getting the type of 'file_' (line 254)
            file__6661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 32), 'file_', False)
            
            # Call to globals(...): (line 254)
            # Processing the call keyword arguments (line 254)
            kwargs_6663 = {}
            # Getting the type of 'globals' (line 254)
            globals_6662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 39), 'globals', False)
            # Calling globals(args, kwargs) (line 254)
            globals_call_result_6664 = invoke(stypy.reporting.localization.Localization(__file__, 254, 39), globals_6662, *[], **kwargs_6663)
            
            
            # Call to locals(...): (line 254)
            # Processing the call keyword arguments (line 254)
            kwargs_6666 = {}
            # Getting the type of 'locals' (line 254)
            locals_6665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 50), 'locals', False)
            # Calling locals(args, kwargs) (line 254)
            locals_call_result_6667 = invoke(stypy.reporting.localization.Localization(__file__, 254, 50), locals_6665, *[], **kwargs_6666)
            
            # Processing the call keyword arguments (line 254)
            kwargs_6668 = {}
            # Getting the type of '__import__' (line 254)
            import___6660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 21), '__import__', False)
            # Calling __import__(args, kwargs) (line 254)
            import___call_result_6669 = invoke(stypy.reporting.localization.Localization(__file__, 254, 21), import___6660, *[file__6661, globals_call_result_6664, locals_call_result_6667], **kwargs_6668)
            
            # Assigning a type to the variable 'module' (line 254)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 12), 'module', import___call_result_6669)
            
            # Assigning a Subscript to a Name (line 255):
            
            # Assigning a Subscript to a Name (line 255):
            
            # Obtaining the type of the subscript
            int_6670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 52), 'int')
            
            # Call to split(...): (line 255)
            # Processing the call arguments (line 255)
            str_6674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 47), 'str', '.')
            # Processing the call keyword arguments (line 255)
            kwargs_6675 = {}
            # Getting the type of 'proxy_obj' (line 255)
            proxy_obj_6671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 26), 'proxy_obj', False)
            # Obtaining the member 'name' of a type (line 255)
            name_6672 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 26), proxy_obj_6671, 'name')
            # Obtaining the member 'split' of a type (line 255)
            split_6673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 26), name_6672, 'split')
            # Calling split(args, kwargs) (line 255)
            split_call_result_6676 = invoke(stypy.reporting.localization.Localization(__file__, 255, 26), split_6673, *[str_6674], **kwargs_6675)
            
            # Obtaining the member '__getitem__' of a type (line 255)
            getitem___6677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 26), split_call_result_6676, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 255)
            subscript_call_result_6678 = invoke(stypy.reporting.localization.Localization(__file__, 255, 26), getitem___6677, int_6670)
            
            # Assigning a type to the variable 'entity_name' (line 255)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 12), 'entity_name', subscript_call_result_6678)
            
            
            # SSA begins for try-except statement (line 256)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
            
            # Assigning a Subscript to a Name (line 260):
            
            # Assigning a Subscript to a Name (line 260):
            
            # Obtaining the type of the subscript
            # Getting the type of 'entity_name' (line 260)
            entity_name_6679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 53), 'entity_name')
            # Getting the type of 'module' (line 260)
            module_6680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 24), 'module')
            # Obtaining the member 'type_rules_of_members' of a type (line 260)
            type_rules_of_members_6681 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 24), module_6680, 'type_rules_of_members')
            # Obtaining the member '__getitem__' of a type (line 260)
            getitem___6682 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 24), type_rules_of_members_6681, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 260)
            subscript_call_result_6683 = invoke(stypy.reporting.localization.Localization(__file__, 260, 24), getitem___6682, entity_name_6679)
            
            # Assigning a type to the variable 'rules' (line 260)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 16), 'rules', subscript_call_result_6683)
            
            # Call to isfunction(...): (line 263)
            # Processing the call arguments (line 263)
            # Getting the type of 'rules' (line 263)
            rules_6686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 38), 'rules', False)
            # Processing the call keyword arguments (line 263)
            kwargs_6687 = {}
            # Getting the type of 'inspect' (line 263)
            inspect_6684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 19), 'inspect', False)
            # Obtaining the member 'isfunction' of a type (line 263)
            isfunction_6685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 19), inspect_6684, 'isfunction')
            # Calling isfunction(args, kwargs) (line 263)
            isfunction_call_result_6688 = invoke(stypy.reporting.localization.Localization(__file__, 263, 19), isfunction_6685, *[rules_6686], **kwargs_6687)
            
            # Testing if the type of an if condition is none (line 263)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 263, 16), isfunction_call_result_6688):
                pass
            else:
                
                # Testing the type of an if condition (line 263)
                if_condition_6689 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 263, 16), isfunction_call_result_6688)
                # Assigning a type to the variable 'if_condition_6689' (line 263)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 16), 'if_condition_6689', if_condition_6689)
                # SSA begins for if statement (line 263)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Name (line 264):
                
                # Assigning a Call to a Name (line 264):
                
                # Call to rules(...): (line 264)
                # Processing the call keyword arguments (line 264)
                kwargs_6691 = {}
                # Getting the type of 'rules' (line 264)
                rules_6690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 28), 'rules', False)
                # Calling rules(args, kwargs) (line 264)
                rules_call_result_6692 = invoke(stypy.reporting.localization.Localization(__file__, 264, 28), rules_6690, *[], **kwargs_6691)
                
                # Assigning a type to the variable 'rules' (line 264)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 20), 'rules', rules_call_result_6692)
                # SSA join for if statement (line 263)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Assigning a Name to a Subscript (line 267):
            
            # Assigning a Name to a Subscript (line 267):
            # Getting the type of 'rules' (line 267)
            rules_6693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 51), 'rules')
            # Getting the type of 'self' (line 267)
            self_6694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 16), 'self')
            # Obtaining the member 'type_rule_cache' of a type (line 267)
            type_rule_cache_6695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 16), self_6694, 'type_rule_cache')
            # Getting the type of 'cache_name' (line 267)
            cache_name_6696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 37), 'cache_name')
            # Storing an element on a container (line 267)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 267, 16), type_rule_cache_6695, (cache_name_6696, rules_6693))
            # SSA branch for the except part of a try statement (line 256)
            # SSA branch for the except '<any exception>' branch of a try statement (line 256)
            module_type_store.open_ssa_branch('except')
            
            # Assigning a Name to a Subscript (line 270):
            
            # Assigning a Name to a Subscript (line 270):
            # Getting the type of 'True' (line 270)
            True_6697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 63), 'True')
            # Getting the type of 'self' (line 270)
            self_6698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 16), 'self')
            # Obtaining the member 'unavailable_type_rule_cache' of a type (line 270)
            unavailable_type_rule_cache_6699 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 16), self_6698, 'unavailable_type_rule_cache')
            # Getting the type of 'cache_name' (line 270)
            cache_name_6700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 49), 'cache_name')
            # Storing an element on a container (line 270)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 270, 16), unavailable_type_rule_cache_6699, (cache_name_6700, True_6697))
            # Getting the type of 'False' (line 271)
            False_6701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 23), 'False')
            # Assigning a type to the variable 'stypy_return_type' (line 271)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 16), 'stypy_return_type', False_6701)
            # SSA join for try-except statement (line 256)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 249)
            module_type_store = module_type_store.join_ssa_context()
            

        
        
        # Evaluating a boolean operation
        # Getting the type of 'parent_exist' (line 273)
        parent_exist_6702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 16), 'parent_exist')
        # Getting the type of 'own_exist' (line 273)
        own_exist_6703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 32), 'own_exist')
        # Applying the binary operator 'or' (line 273)
        result_or_keyword_6704 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 16), 'or', parent_exist_6702, own_exist_6703)
        
        # Applying the 'not' unary operator (line 273)
        result_not__6705 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 11), 'not', result_or_keyword_6704)
        
        # Testing if the type of an if condition is none (line 273)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 273, 8), result_not__6705):
            pass
        else:
            
            # Testing the type of an if condition (line 273)
            if_condition_6706 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 273, 8), result_not__6705)
            # Assigning a type to the variable 'if_condition_6706' (line 273)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 8), 'if_condition_6706', if_condition_6706)
            # SSA begins for if statement (line 273)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'proxy_obj' (line 274)
            proxy_obj_6707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 15), 'proxy_obj')
            # Obtaining the member 'name' of a type (line 274)
            name_6708 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 15), proxy_obj_6707, 'name')
            # Getting the type of 'self' (line 274)
            self_6709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 37), 'self')
            # Obtaining the member 'unavailable_type_rule_cache' of a type (line 274)
            unavailable_type_rule_cache_6710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 37), self_6709, 'unavailable_type_rule_cache')
            # Applying the binary operator 'notin' (line 274)
            result_contains_6711 = python_operator(stypy.reporting.localization.Localization(__file__, 274, 15), 'notin', name_6708, unavailable_type_rule_cache_6710)
            
            # Testing if the type of an if condition is none (line 274)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 274, 12), result_contains_6711):
                pass
            else:
                
                # Testing the type of an if condition (line 274)
                if_condition_6712 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 274, 12), result_contains_6711)
                # Assigning a type to the variable 'if_condition_6712' (line 274)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 12), 'if_condition_6712', if_condition_6712)
                # SSA begins for if statement (line 274)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Name to a Subscript (line 276):
                
                # Assigning a Name to a Subscript (line 276):
                # Getting the type of 'True' (line 276)
                True_6713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 63), 'True')
                # Getting the type of 'self' (line 276)
                self_6714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 16), 'self')
                # Obtaining the member 'unavailable_type_rule_cache' of a type (line 276)
                unavailable_type_rule_cache_6715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 16), self_6714, 'unavailable_type_rule_cache')
                # Getting the type of 'cache_name' (line 276)
                cache_name_6716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 49), 'cache_name')
                # Storing an element on a container (line 276)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 276, 16), unavailable_type_rule_cache_6715, (cache_name_6716, True_6713))
                # SSA join for if statement (line 274)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 273)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Evaluating a boolean operation
        # Getting the type of 'parent_exist' (line 278)
        parent_exist_6717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 15), 'parent_exist')
        # Getting the type of 'own_exist' (line 278)
        own_exist_6718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 31), 'own_exist')
        # Applying the binary operator 'or' (line 278)
        result_or_keyword_6719 = python_operator(stypy.reporting.localization.Localization(__file__, 278, 15), 'or', parent_exist_6717, own_exist_6718)
        
        # Assigning a type to the variable 'stypy_return_type' (line 278)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 8), 'stypy_return_type', result_or_keyword_6719)
        
        # ################# End of 'applies_to(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'applies_to' in the type store
        # Getting the type of 'stypy_return_type' (line 175)
        stypy_return_type_6720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6720)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'applies_to'
        return stypy_return_type_6720


    @norecursion
    def __get_rules_and_name(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__get_rules_and_name'
        module_type_store = module_type_store.open_function_context('__get_rules_and_name', 280, 4, False)
        # Assigning a type to the variable 'self' (line 281)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeRuleCallHandler.__get_rules_and_name.__dict__.__setitem__('stypy_localization', localization)
        TypeRuleCallHandler.__get_rules_and_name.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeRuleCallHandler.__get_rules_and_name.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeRuleCallHandler.__get_rules_and_name.__dict__.__setitem__('stypy_function_name', 'TypeRuleCallHandler.__get_rules_and_name')
        TypeRuleCallHandler.__get_rules_and_name.__dict__.__setitem__('stypy_param_names_list', ['entity_name', 'parent_name'])
        TypeRuleCallHandler.__get_rules_and_name.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeRuleCallHandler.__get_rules_and_name.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeRuleCallHandler.__get_rules_and_name.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeRuleCallHandler.__get_rules_and_name.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeRuleCallHandler.__get_rules_and_name.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeRuleCallHandler.__get_rules_and_name.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeRuleCallHandler.__get_rules_and_name', ['entity_name', 'parent_name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__get_rules_and_name', localization, ['entity_name', 'parent_name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__get_rules_and_name(...)' code ##################

        str_6721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, (-1)), 'str', '\n        Obtain a member name and its type rules\n        :param entity_name: Entity name\n        :param parent_name: Entity container name\n        :return: tuple (name, rules tied to this name)\n        ')
        
        # Getting the type of 'entity_name' (line 287)
        entity_name_6722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 11), 'entity_name')
        # Getting the type of 'self' (line 287)
        self_6723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 26), 'self')
        # Obtaining the member 'type_rule_cache' of a type (line 287)
        type_rule_cache_6724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 26), self_6723, 'type_rule_cache')
        # Applying the binary operator 'in' (line 287)
        result_contains_6725 = python_operator(stypy.reporting.localization.Localization(__file__, 287, 11), 'in', entity_name_6722, type_rule_cache_6724)
        
        # Testing if the type of an if condition is none (line 287)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 287, 8), result_contains_6725):
            pass
        else:
            
            # Testing the type of an if condition (line 287)
            if_condition_6726 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 287, 8), result_contains_6725)
            # Assigning a type to the variable 'if_condition_6726' (line 287)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'if_condition_6726', if_condition_6726)
            # SSA begins for if statement (line 287)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Name to a Name (line 288):
            
            # Assigning a Name to a Name (line 288):
            # Getting the type of 'entity_name' (line 288)
            entity_name_6727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 19), 'entity_name')
            # Assigning a type to the variable 'name' (line 288)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 12), 'name', entity_name_6727)
            
            # Assigning a Subscript to a Name (line 289):
            
            # Assigning a Subscript to a Name (line 289):
            
            # Obtaining the type of the subscript
            # Getting the type of 'entity_name' (line 289)
            entity_name_6728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 41), 'entity_name')
            # Getting the type of 'self' (line 289)
            self_6729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 20), 'self')
            # Obtaining the member 'type_rule_cache' of a type (line 289)
            type_rule_cache_6730 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 20), self_6729, 'type_rule_cache')
            # Obtaining the member '__getitem__' of a type (line 289)
            getitem___6731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 20), type_rule_cache_6730, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 289)
            subscript_call_result_6732 = invoke(stypy.reporting.localization.Localization(__file__, 289, 20), getitem___6731, entity_name_6728)
            
            # Assigning a type to the variable 'rules' (line 289)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 12), 'rules', subscript_call_result_6732)
            
            # Obtaining an instance of the builtin type 'tuple' (line 291)
            tuple_6733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 19), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 291)
            # Adding element type (line 291)
            # Getting the type of 'name' (line 291)
            name_6734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 19), 'name')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 19), tuple_6733, name_6734)
            # Adding element type (line 291)
            # Getting the type of 'rules' (line 291)
            rules_6735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 25), 'rules')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 19), tuple_6733, rules_6735)
            
            # Assigning a type to the variable 'stypy_return_type' (line 291)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 12), 'stypy_return_type', tuple_6733)
            # SSA join for if statement (line 287)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 'parent_name' (line 293)
        parent_name_6736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 11), 'parent_name')
        # Getting the type of 'self' (line 293)
        self_6737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 26), 'self')
        # Obtaining the member 'type_rule_cache' of a type (line 293)
        type_rule_cache_6738 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 26), self_6737, 'type_rule_cache')
        # Applying the binary operator 'in' (line 293)
        result_contains_6739 = python_operator(stypy.reporting.localization.Localization(__file__, 293, 11), 'in', parent_name_6736, type_rule_cache_6738)
        
        # Testing if the type of an if condition is none (line 293)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 293, 8), result_contains_6739):
            pass
        else:
            
            # Testing the type of an if condition (line 293)
            if_condition_6740 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 293, 8), result_contains_6739)
            # Assigning a type to the variable 'if_condition_6740' (line 293)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 8), 'if_condition_6740', if_condition_6740)
            # SSA begins for if statement (line 293)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Name to a Name (line 294):
            
            # Assigning a Name to a Name (line 294):
            # Getting the type of 'parent_name' (line 294)
            parent_name_6741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 19), 'parent_name')
            # Assigning a type to the variable 'name' (line 294)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 12), 'name', parent_name_6741)
            
            # Assigning a Subscript to a Name (line 295):
            
            # Assigning a Subscript to a Name (line 295):
            
            # Obtaining the type of the subscript
            # Getting the type of 'parent_name' (line 295)
            parent_name_6742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 41), 'parent_name')
            # Getting the type of 'self' (line 295)
            self_6743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 20), 'self')
            # Obtaining the member 'type_rule_cache' of a type (line 295)
            type_rule_cache_6744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 20), self_6743, 'type_rule_cache')
            # Obtaining the member '__getitem__' of a type (line 295)
            getitem___6745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 20), type_rule_cache_6744, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 295)
            subscript_call_result_6746 = invoke(stypy.reporting.localization.Localization(__file__, 295, 20), getitem___6745, parent_name_6742)
            
            # Assigning a type to the variable 'rules' (line 295)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 12), 'rules', subscript_call_result_6746)
            
            # Obtaining an instance of the builtin type 'tuple' (line 297)
            tuple_6747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 19), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 297)
            # Adding element type (line 297)
            # Getting the type of 'name' (line 297)
            name_6748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 19), 'name')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 297, 19), tuple_6747, name_6748)
            # Adding element type (line 297)
            # Getting the type of 'rules' (line 297)
            rules_6749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 25), 'rules')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 297, 19), tuple_6747, rules_6749)
            
            # Assigning a type to the variable 'stypy_return_type' (line 297)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 12), 'stypy_return_type', tuple_6747)
            # SSA join for if statement (line 293)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of '__get_rules_and_name(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__get_rules_and_name' in the type store
        # Getting the type of 'stypy_return_type' (line 280)
        stypy_return_type_6750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6750)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__get_rules_and_name'
        return stypy_return_type_6750


    @staticmethod
    @norecursion
    def __format_admitted_params(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__format_admitted_params'
        module_type_store = module_type_store.open_function_context('__format_admitted_params', 299, 4, False)
        
        # Passed parameters checking function
        TypeRuleCallHandler.__format_admitted_params.__dict__.__setitem__('stypy_localization', localization)
        TypeRuleCallHandler.__format_admitted_params.__dict__.__setitem__('stypy_type_of_self', None)
        TypeRuleCallHandler.__format_admitted_params.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeRuleCallHandler.__format_admitted_params.__dict__.__setitem__('stypy_function_name', '__format_admitted_params')
        TypeRuleCallHandler.__format_admitted_params.__dict__.__setitem__('stypy_param_names_list', ['name', 'rules', 'arguments', 'call_arity'])
        TypeRuleCallHandler.__format_admitted_params.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeRuleCallHandler.__format_admitted_params.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeRuleCallHandler.__format_admitted_params.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeRuleCallHandler.__format_admitted_params.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeRuleCallHandler.__format_admitted_params.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeRuleCallHandler.__format_admitted_params.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, None, module_type_store, '__format_admitted_params', ['name', 'rules', 'arguments', 'call_arity'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__format_admitted_params', localization, ['rules', 'arguments', 'call_arity'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__format_admitted_params(...)' code ##################

        str_6751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, (-1)), 'str', '\n        Pretty-print error message when no type rule for the member matches with the arguments of the call\n        :param name: Member name\n        :param rules: Rules tied to this member name\n        :param arguments: Call arguments\n        :param call_arity: Call arity\n        :return:\n        ')
        
        # Assigning a BinOp to a Name (line 309):
        
        # Assigning a BinOp to a Name (line 309):
        
        # Obtaining an instance of the builtin type 'list' (line 309)
        list_6752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 309)
        # Adding element type (line 309)
        str_6753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 23), 'str', '')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 309, 22), list_6752, str_6753)
        
        # Getting the type of 'call_arity' (line 309)
        call_arity_6754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 29), 'call_arity')
        # Applying the binary operator '*' (line 309)
        result_mul_6755 = python_operator(stypy.reporting.localization.Localization(__file__, 309, 22), '*', list_6752, call_arity_6754)
        
        # Assigning a type to the variable 'params_strs' (line 309)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 8), 'params_strs', result_mul_6755)
        
        # Assigning a Name to a Name (line 310):
        
        # Assigning a Name to a Name (line 310):
        # Getting the type of 'True' (line 310)
        True_6756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 21), 'True')
        # Assigning a type to the variable 'first_rule' (line 310)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 8), 'first_rule', True_6756)
        
        # Assigning a List to a Name (line 311):
        
        # Assigning a List to a Name (line 311):
        
        # Obtaining an instance of the builtin type 'list' (line 311)
        list_6757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 311)
        
        # Assigning a type to the variable 'arities' (line 311)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 8), 'arities', list_6757)
        
        # Assigning a Name to a Name (line 314):
        
        # Assigning a Name to a Name (line 314):
        # Getting the type of 'False' (line 314)
        False_6758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 38), 'False')
        # Assigning a type to the variable 'rules_with_enough_arguments' (line 314)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 8), 'rules_with_enough_arguments', False_6758)
        
        # Getting the type of 'rules' (line 315)
        rules_6759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 46), 'rules')
        # Assigning a type to the variable 'rules_6759' (line 315)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 8), 'rules_6759', rules_6759)
        # Testing if the for loop is going to be iterated (line 315)
        # Testing the type of a for loop iterable (line 315)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 315, 8), rules_6759)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 315, 8), rules_6759):
            # Getting the type of the for loop variable (line 315)
            for_loop_var_6760 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 315, 8), rules_6759)
            # Assigning a type to the variable 'params_in_rules' (line 315)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 8), 'params_in_rules', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 315, 8), for_loop_var_6760, 2, 0))
            # Assigning a type to the variable 'return_type' (line 315)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 8), 'return_type', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 315, 8), for_loop_var_6760, 2, 1))
            # SSA begins for a for statement (line 315)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Call to a Name (line 316):
            
            # Assigning a Call to a Name (line 316):
            
            # Call to len(...): (line 316)
            # Processing the call arguments (line 316)
            # Getting the type of 'params_in_rules' (line 316)
            params_in_rules_6762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 27), 'params_in_rules', False)
            # Processing the call keyword arguments (line 316)
            kwargs_6763 = {}
            # Getting the type of 'len' (line 316)
            len_6761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 23), 'len', False)
            # Calling len(args, kwargs) (line 316)
            len_call_result_6764 = invoke(stypy.reporting.localization.Localization(__file__, 316, 23), len_6761, *[params_in_rules_6762], **kwargs_6763)
            
            # Assigning a type to the variable 'rule_len' (line 316)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 12), 'rule_len', len_call_result_6764)
            
            # Getting the type of 'rule_len' (line 317)
            rule_len_6765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 15), 'rule_len')
            # Getting the type of 'arities' (line 317)
            arities_6766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 31), 'arities')
            # Applying the binary operator 'notin' (line 317)
            result_contains_6767 = python_operator(stypy.reporting.localization.Localization(__file__, 317, 15), 'notin', rule_len_6765, arities_6766)
            
            # Testing if the type of an if condition is none (line 317)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 317, 12), result_contains_6767):
                pass
            else:
                
                # Testing the type of an if condition (line 317)
                if_condition_6768 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 317, 12), result_contains_6767)
                # Assigning a type to the variable 'if_condition_6768' (line 317)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 12), 'if_condition_6768', if_condition_6768)
                # SSA begins for if statement (line 317)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to append(...): (line 318)
                # Processing the call arguments (line 318)
                # Getting the type of 'rule_len' (line 318)
                rule_len_6771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 31), 'rule_len', False)
                # Processing the call keyword arguments (line 318)
                kwargs_6772 = {}
                # Getting the type of 'arities' (line 318)
                arities_6769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 16), 'arities', False)
                # Obtaining the member 'append' of a type (line 318)
                append_6770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 16), arities_6769, 'append')
                # Calling append(args, kwargs) (line 318)
                append_call_result_6773 = invoke(stypy.reporting.localization.Localization(__file__, 318, 16), append_6770, *[rule_len_6771], **kwargs_6772)
                
                # SSA join for if statement (line 317)
                module_type_store = module_type_store.join_ssa_context()
                

            
            
            # Call to len(...): (line 320)
            # Processing the call arguments (line 320)
            # Getting the type of 'params_in_rules' (line 320)
            params_in_rules_6775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 19), 'params_in_rules', False)
            # Processing the call keyword arguments (line 320)
            kwargs_6776 = {}
            # Getting the type of 'len' (line 320)
            len_6774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 15), 'len', False)
            # Calling len(args, kwargs) (line 320)
            len_call_result_6777 = invoke(stypy.reporting.localization.Localization(__file__, 320, 15), len_6774, *[params_in_rules_6775], **kwargs_6776)
            
            # Getting the type of 'call_arity' (line 320)
            call_arity_6778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 39), 'call_arity')
            # Applying the binary operator '==' (line 320)
            result_eq_6779 = python_operator(stypy.reporting.localization.Localization(__file__, 320, 15), '==', len_call_result_6777, call_arity_6778)
            
            # Testing if the type of an if condition is none (line 320)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 320, 12), result_eq_6779):
                pass
            else:
                
                # Testing the type of an if condition (line 320)
                if_condition_6780 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 320, 12), result_eq_6779)
                # Assigning a type to the variable 'if_condition_6780' (line 320)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 12), 'if_condition_6780', if_condition_6780)
                # SSA begins for if statement (line 320)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Name to a Name (line 321):
                
                # Assigning a Name to a Name (line 321):
                # Getting the type of 'True' (line 321)
                True_6781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 46), 'True')
                # Assigning a type to the variable 'rules_with_enough_arguments' (line 321)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 16), 'rules_with_enough_arguments', True_6781)
                # SSA join for if statement (line 320)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Getting the type of 'rules_with_enough_arguments' (line 323)
        rules_with_enough_arguments_6782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 15), 'rules_with_enough_arguments')
        # Applying the 'not' unary operator (line 323)
        result_not__6783 = python_operator(stypy.reporting.localization.Localization(__file__, 323, 11), 'not', rules_with_enough_arguments_6782)
        
        # Testing if the type of an if condition is none (line 323)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 323, 8), result_not__6783):
            pass
        else:
            
            # Testing the type of an if condition (line 323)
            if_condition_6784 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 323, 8), result_not__6783)
            # Assigning a type to the variable 'if_condition_6784' (line 323)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 8), 'if_condition_6784', if_condition_6784)
            # SSA begins for if statement (line 323)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Str to a Name (line 324):
            
            # Assigning a Str to a Name (line 324):
            str_6785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 26), 'str', '')
            # Assigning a type to the variable 'str_arities' (line 324)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 12), 'str_arities', str_6785)
            
            
            # Call to range(...): (line 325)
            # Processing the call arguments (line 325)
            
            # Call to len(...): (line 325)
            # Processing the call arguments (line 325)
            # Getting the type of 'arities' (line 325)
            arities_6788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 31), 'arities', False)
            # Processing the call keyword arguments (line 325)
            kwargs_6789 = {}
            # Getting the type of 'len' (line 325)
            len_6787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 27), 'len', False)
            # Calling len(args, kwargs) (line 325)
            len_call_result_6790 = invoke(stypy.reporting.localization.Localization(__file__, 325, 27), len_6787, *[arities_6788], **kwargs_6789)
            
            # Processing the call keyword arguments (line 325)
            kwargs_6791 = {}
            # Getting the type of 'range' (line 325)
            range_6786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 21), 'range', False)
            # Calling range(args, kwargs) (line 325)
            range_call_result_6792 = invoke(stypy.reporting.localization.Localization(__file__, 325, 21), range_6786, *[len_call_result_6790], **kwargs_6791)
            
            # Assigning a type to the variable 'range_call_result_6792' (line 325)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 12), 'range_call_result_6792', range_call_result_6792)
            # Testing if the for loop is going to be iterated (line 325)
            # Testing the type of a for loop iterable (line 325)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 325, 12), range_call_result_6792)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 325, 12), range_call_result_6792):
                # Getting the type of the for loop variable (line 325)
                for_loop_var_6793 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 325, 12), range_call_result_6792)
                # Assigning a type to the variable 'i' (line 325)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 12), 'i', for_loop_var_6793)
                # SSA begins for a for statement (line 325)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Getting the type of 'str_arities' (line 326)
                str_arities_6794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 16), 'str_arities')
                
                # Call to str(...): (line 326)
                # Processing the call arguments (line 326)
                
                # Obtaining the type of the subscript
                # Getting the type of 'i' (line 326)
                i_6796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 43), 'i', False)
                # Getting the type of 'arities' (line 326)
                arities_6797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 35), 'arities', False)
                # Obtaining the member '__getitem__' of a type (line 326)
                getitem___6798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 35), arities_6797, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 326)
                subscript_call_result_6799 = invoke(stypy.reporting.localization.Localization(__file__, 326, 35), getitem___6798, i_6796)
                
                # Processing the call keyword arguments (line 326)
                kwargs_6800 = {}
                # Getting the type of 'str' (line 326)
                str_6795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 31), 'str', False)
                # Calling str(args, kwargs) (line 326)
                str_call_result_6801 = invoke(stypy.reporting.localization.Localization(__file__, 326, 31), str_6795, *[subscript_call_result_6799], **kwargs_6800)
                
                # Applying the binary operator '+=' (line 326)
                result_iadd_6802 = python_operator(stypy.reporting.localization.Localization(__file__, 326, 16), '+=', str_arities_6794, str_call_result_6801)
                # Assigning a type to the variable 'str_arities' (line 326)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 16), 'str_arities', result_iadd_6802)
                
                
                
                # Call to len(...): (line 327)
                # Processing the call arguments (line 327)
                # Getting the type of 'arities' (line 327)
                arities_6804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 23), 'arities', False)
                # Processing the call keyword arguments (line 327)
                kwargs_6805 = {}
                # Getting the type of 'len' (line 327)
                len_6803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 19), 'len', False)
                # Calling len(args, kwargs) (line 327)
                len_call_result_6806 = invoke(stypy.reporting.localization.Localization(__file__, 327, 19), len_6803, *[arities_6804], **kwargs_6805)
                
                int_6807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 34), 'int')
                # Applying the binary operator '>' (line 327)
                result_gt_6808 = python_operator(stypy.reporting.localization.Localization(__file__, 327, 19), '>', len_call_result_6806, int_6807)
                
                # Testing if the type of an if condition is none (line 327)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 327, 16), result_gt_6808):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 327)
                    if_condition_6809 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 327, 16), result_gt_6808)
                    # Assigning a type to the variable 'if_condition_6809' (line 327)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 16), 'if_condition_6809', if_condition_6809)
                    # SSA begins for if statement (line 327)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Getting the type of 'i' (line 328)
                    i_6810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 23), 'i')
                    
                    # Call to len(...): (line 328)
                    # Processing the call arguments (line 328)
                    # Getting the type of 'arities' (line 328)
                    arities_6812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 32), 'arities', False)
                    # Processing the call keyword arguments (line 328)
                    kwargs_6813 = {}
                    # Getting the type of 'len' (line 328)
                    len_6811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 28), 'len', False)
                    # Calling len(args, kwargs) (line 328)
                    len_call_result_6814 = invoke(stypy.reporting.localization.Localization(__file__, 328, 28), len_6811, *[arities_6812], **kwargs_6813)
                    
                    int_6815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 43), 'int')
                    # Applying the binary operator '-' (line 328)
                    result_sub_6816 = python_operator(stypy.reporting.localization.Localization(__file__, 328, 28), '-', len_call_result_6814, int_6815)
                    
                    # Applying the binary operator '==' (line 328)
                    result_eq_6817 = python_operator(stypy.reporting.localization.Localization(__file__, 328, 23), '==', i_6810, result_sub_6816)
                    
                    # Testing if the type of an if condition is none (line 328)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 328, 20), result_eq_6817):
                        
                        # Getting the type of 'str_arities' (line 331)
                        str_arities_6822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 24), 'str_arities')
                        str_6823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 39), 'str', ', ')
                        # Applying the binary operator '+=' (line 331)
                        result_iadd_6824 = python_operator(stypy.reporting.localization.Localization(__file__, 331, 24), '+=', str_arities_6822, str_6823)
                        # Assigning a type to the variable 'str_arities' (line 331)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 24), 'str_arities', result_iadd_6824)
                        
                    else:
                        
                        # Testing the type of an if condition (line 328)
                        if_condition_6818 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 328, 20), result_eq_6817)
                        # Assigning a type to the variable 'if_condition_6818' (line 328)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 20), 'if_condition_6818', if_condition_6818)
                        # SSA begins for if statement (line 328)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Getting the type of 'str_arities' (line 329)
                        str_arities_6819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 24), 'str_arities')
                        str_6820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 39), 'str', ' or ')
                        # Applying the binary operator '+=' (line 329)
                        result_iadd_6821 = python_operator(stypy.reporting.localization.Localization(__file__, 329, 24), '+=', str_arities_6819, str_6820)
                        # Assigning a type to the variable 'str_arities' (line 329)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 24), 'str_arities', result_iadd_6821)
                        
                        # SSA branch for the else part of an if statement (line 328)
                        module_type_store.open_ssa_branch('else')
                        
                        # Getting the type of 'str_arities' (line 331)
                        str_arities_6822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 24), 'str_arities')
                        str_6823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 39), 'str', ', ')
                        # Applying the binary operator '+=' (line 331)
                        result_iadd_6824 = python_operator(stypy.reporting.localization.Localization(__file__, 331, 24), '+=', str_arities_6822, str_6823)
                        # Assigning a type to the variable 'str_arities' (line 331)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 24), 'str_arities', result_iadd_6824)
                        
                        # SSA join for if statement (line 328)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for if statement (line 327)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            
            # Call to format(...): (line 332)
            # Processing the call arguments (line 332)
            # Getting the type of 'call_arity' (line 333)
            call_arity_6827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 16), 'call_arity', False)
            # Getting the type of 'str_arities' (line 334)
            str_arities_6828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 16), 'str_arities', False)
            # Processing the call keyword arguments (line 332)
            kwargs_6829 = {}
            str_6825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 19), 'str', 'The invocation was performed with {0} argument(s), but only {1} argument(s) are accepted')
            # Obtaining the member 'format' of a type (line 332)
            format_6826 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 19), str_6825, 'format')
            # Calling format(args, kwargs) (line 332)
            format_call_result_6830 = invoke(stypy.reporting.localization.Localization(__file__, 332, 19), format_6826, *[call_arity_6827, str_arities_6828], **kwargs_6829)
            
            # Assigning a type to the variable 'stypy_return_type' (line 332)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 12), 'stypy_return_type', format_call_result_6830)
            # SSA join for if statement (line 323)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 'rules' (line 336)
        rules_6831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 46), 'rules')
        # Assigning a type to the variable 'rules_6831' (line 336)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 8), 'rules_6831', rules_6831)
        # Testing if the for loop is going to be iterated (line 336)
        # Testing the type of a for loop iterable (line 336)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 336, 8), rules_6831)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 336, 8), rules_6831):
            # Getting the type of the for loop variable (line 336)
            for_loop_var_6832 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 336, 8), rules_6831)
            # Assigning a type to the variable 'params_in_rules' (line 336)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 8), 'params_in_rules', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 336, 8), for_loop_var_6832, 2, 0))
            # Assigning a type to the variable 'return_type' (line 336)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 8), 'return_type', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 336, 8), for_loop_var_6832, 2, 1))
            # SSA begins for a for statement (line 336)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Call to len(...): (line 337)
            # Processing the call arguments (line 337)
            # Getting the type of 'params_in_rules' (line 337)
            params_in_rules_6834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 19), 'params_in_rules', False)
            # Processing the call keyword arguments (line 337)
            kwargs_6835 = {}
            # Getting the type of 'len' (line 337)
            len_6833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 15), 'len', False)
            # Calling len(args, kwargs) (line 337)
            len_call_result_6836 = invoke(stypy.reporting.localization.Localization(__file__, 337, 15), len_6833, *[params_in_rules_6834], **kwargs_6835)
            
            # Getting the type of 'call_arity' (line 337)
            call_arity_6837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 39), 'call_arity')
            # Applying the binary operator '==' (line 337)
            result_eq_6838 = python_operator(stypy.reporting.localization.Localization(__file__, 337, 15), '==', len_call_result_6836, call_arity_6837)
            
            # Testing if the type of an if condition is none (line 337)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 337, 12), result_eq_6838):
                pass
            else:
                
                # Testing the type of an if condition (line 337)
                if_condition_6839 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 337, 12), result_eq_6838)
                # Assigning a type to the variable 'if_condition_6839' (line 337)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 12), 'if_condition_6839', if_condition_6839)
                # SSA begins for if statement (line 337)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                
                # Call to range(...): (line 338)
                # Processing the call arguments (line 338)
                # Getting the type of 'call_arity' (line 338)
                call_arity_6841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 31), 'call_arity', False)
                # Processing the call keyword arguments (line 338)
                kwargs_6842 = {}
                # Getting the type of 'range' (line 338)
                range_6840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 25), 'range', False)
                # Calling range(args, kwargs) (line 338)
                range_call_result_6843 = invoke(stypy.reporting.localization.Localization(__file__, 338, 25), range_6840, *[call_arity_6841], **kwargs_6842)
                
                # Assigning a type to the variable 'range_call_result_6843' (line 338)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 16), 'range_call_result_6843', range_call_result_6843)
                # Testing if the for loop is going to be iterated (line 338)
                # Testing the type of a for loop iterable (line 338)
                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 338, 16), range_call_result_6843)

                if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 338, 16), range_call_result_6843):
                    # Getting the type of the for loop variable (line 338)
                    for_loop_var_6844 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 338, 16), range_call_result_6843)
                    # Assigning a type to the variable 'i' (line 338)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 16), 'i', for_loop_var_6844)
                    # SSA begins for a for statement (line 338)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                    
                    # Assigning a Call to a Name (line 339):
                    
                    # Assigning a Call to a Name (line 339):
                    
                    # Call to str(...): (line 339)
                    # Processing the call arguments (line 339)
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'i' (line 339)
                    i_6846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 48), 'i', False)
                    # Getting the type of 'params_in_rules' (line 339)
                    params_in_rules_6847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 32), 'params_in_rules', False)
                    # Obtaining the member '__getitem__' of a type (line 339)
                    getitem___6848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 32), params_in_rules_6847, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 339)
                    subscript_call_result_6849 = invoke(stypy.reporting.localization.Localization(__file__, 339, 32), getitem___6848, i_6846)
                    
                    # Processing the call keyword arguments (line 339)
                    kwargs_6850 = {}
                    # Getting the type of 'str' (line 339)
                    str_6845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 28), 'str', False)
                    # Calling str(args, kwargs) (line 339)
                    str_call_result_6851 = invoke(stypy.reporting.localization.Localization(__file__, 339, 28), str_6845, *[subscript_call_result_6849], **kwargs_6850)
                    
                    # Assigning a type to the variable 'value' (line 339)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 20), 'value', str_call_result_6851)
                    
                    # Getting the type of 'value' (line 340)
                    value_6852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 23), 'value')
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'i' (line 340)
                    i_6853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 48), 'i')
                    # Getting the type of 'params_strs' (line 340)
                    params_strs_6854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 36), 'params_strs')
                    # Obtaining the member '__getitem__' of a type (line 340)
                    getitem___6855 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 36), params_strs_6854, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 340)
                    subscript_call_result_6856 = invoke(stypy.reporting.localization.Localization(__file__, 340, 36), getitem___6855, i_6853)
                    
                    # Applying the binary operator 'notin' (line 340)
                    result_contains_6857 = python_operator(stypy.reporting.localization.Localization(__file__, 340, 23), 'notin', value_6852, subscript_call_result_6856)
                    
                    # Testing if the type of an if condition is none (line 340)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 340, 20), result_contains_6857):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 340)
                        if_condition_6858 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 340, 20), result_contains_6857)
                        # Assigning a type to the variable 'if_condition_6858' (line 340)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 20), 'if_condition_6858', if_condition_6858)
                        # SSA begins for if statement (line 340)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Getting the type of 'first_rule' (line 341)
                        first_rule_6859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 31), 'first_rule')
                        # Applying the 'not' unary operator (line 341)
                        result_not__6860 = python_operator(stypy.reporting.localization.Localization(__file__, 341, 27), 'not', first_rule_6859)
                        
                        # Testing if the type of an if condition is none (line 341)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 341, 24), result_not__6860):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 341)
                            if_condition_6861 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 341, 24), result_not__6860)
                            # Assigning a type to the variable 'if_condition_6861' (line 341)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 24), 'if_condition_6861', if_condition_6861)
                            # SSA begins for if statement (line 341)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Getting the type of 'params_strs' (line 342)
                            params_strs_6862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 28), 'params_strs')
                            
                            # Obtaining the type of the subscript
                            # Getting the type of 'i' (line 342)
                            i_6863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 40), 'i')
                            # Getting the type of 'params_strs' (line 342)
                            params_strs_6864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 28), 'params_strs')
                            # Obtaining the member '__getitem__' of a type (line 342)
                            getitem___6865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 28), params_strs_6864, '__getitem__')
                            # Calling the subscript (__getitem__) to obtain the elements type (line 342)
                            subscript_call_result_6866 = invoke(stypy.reporting.localization.Localization(__file__, 342, 28), getitem___6865, i_6863)
                            
                            str_6867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 46), 'str', ' \\/ ')
                            # Applying the binary operator '+=' (line 342)
                            result_iadd_6868 = python_operator(stypy.reporting.localization.Localization(__file__, 342, 28), '+=', subscript_call_result_6866, str_6867)
                            # Getting the type of 'params_strs' (line 342)
                            params_strs_6869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 28), 'params_strs')
                            # Getting the type of 'i' (line 342)
                            i_6870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 40), 'i')
                            # Storing an element on a container (line 342)
                            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 342, 28), params_strs_6869, (i_6870, result_iadd_6868))
                            
                            # SSA join for if statement (line 341)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        
                        # Getting the type of 'params_strs' (line 343)
                        params_strs_6871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 24), 'params_strs')
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'i' (line 343)
                        i_6872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 36), 'i')
                        # Getting the type of 'params_strs' (line 343)
                        params_strs_6873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 24), 'params_strs')
                        # Obtaining the member '__getitem__' of a type (line 343)
                        getitem___6874 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 24), params_strs_6873, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 343)
                        subscript_call_result_6875 = invoke(stypy.reporting.localization.Localization(__file__, 343, 24), getitem___6874, i_6872)
                        
                        # Getting the type of 'value' (line 343)
                        value_6876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 42), 'value')
                        # Applying the binary operator '+=' (line 343)
                        result_iadd_6877 = python_operator(stypy.reporting.localization.Localization(__file__, 343, 24), '+=', subscript_call_result_6875, value_6876)
                        # Getting the type of 'params_strs' (line 343)
                        params_strs_6878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 24), 'params_strs')
                        # Getting the type of 'i' (line 343)
                        i_6879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 36), 'i')
                        # Storing an element on a container (line 343)
                        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 343, 24), params_strs_6878, (i_6879, result_iadd_6877))
                        
                        # SSA join for if statement (line 340)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for a for statement
                    module_type_store = module_type_store.join_ssa_context()

                
                
                # Assigning a Name to a Name (line 345):
                
                # Assigning a Name to a Name (line 345):
                # Getting the type of 'False' (line 345)
                False_6880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 29), 'False')
                # Assigning a type to the variable 'first_rule' (line 345)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 16), 'first_rule', False_6880)
                # SSA join for if statement (line 337)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Assigning a Str to a Name (line 347):
        
        # Assigning a Str to a Name (line 347):
        str_6881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 16), 'str', '')
        # Assigning a type to the variable 'repr_' (line 347)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 8), 'repr_', str_6881)
        
        # Getting the type of 'params_strs' (line 348)
        params_strs_6882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 20), 'params_strs')
        # Assigning a type to the variable 'params_strs_6882' (line 348)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 8), 'params_strs_6882', params_strs_6882)
        # Testing if the for loop is going to be iterated (line 348)
        # Testing the type of a for loop iterable (line 348)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 348, 8), params_strs_6882)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 348, 8), params_strs_6882):
            # Getting the type of the for loop variable (line 348)
            for_loop_var_6883 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 348, 8), params_strs_6882)
            # Assigning a type to the variable 'str_' (line 348)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 8), 'str_', for_loop_var_6883)
            # SSA begins for a for statement (line 348)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'repr_' (line 349)
            repr__6884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 12), 'repr_')
            # Getting the type of 'str_' (line 349)
            str__6885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 21), 'str_')
            str_6886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 28), 'str', ', ')
            # Applying the binary operator '+' (line 349)
            result_add_6887 = python_operator(stypy.reporting.localization.Localization(__file__, 349, 21), '+', str__6885, str_6886)
            
            # Applying the binary operator '+=' (line 349)
            result_iadd_6888 = python_operator(stypy.reporting.localization.Localization(__file__, 349, 12), '+=', repr__6884, result_add_6887)
            # Assigning a type to the variable 'repr_' (line 349)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 12), 'repr_', result_iadd_6888)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 'name' (line 351)
        name_6889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 15), 'name')
        str_6890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 22), 'str', '(')
        # Applying the binary operator '+' (line 351)
        result_add_6891 = python_operator(stypy.reporting.localization.Localization(__file__, 351, 15), '+', name_6889, str_6890)
        
        
        # Obtaining the type of the subscript
        int_6892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 35), 'int')
        slice_6893 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 351, 28), None, int_6892, None)
        # Getting the type of 'repr_' (line 351)
        repr__6894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 28), 'repr_')
        # Obtaining the member '__getitem__' of a type (line 351)
        getitem___6895 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 28), repr__6894, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 351)
        subscript_call_result_6896 = invoke(stypy.reporting.localization.Localization(__file__, 351, 28), getitem___6895, slice_6893)
        
        # Applying the binary operator '+' (line 351)
        result_add_6897 = python_operator(stypy.reporting.localization.Localization(__file__, 351, 26), '+', result_add_6891, subscript_call_result_6896)
        
        str_6898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 41), 'str', ') expected')
        # Applying the binary operator '+' (line 351)
        result_add_6899 = python_operator(stypy.reporting.localization.Localization(__file__, 351, 39), '+', result_add_6897, str_6898)
        
        # Assigning a type to the variable 'stypy_return_type' (line 351)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 8), 'stypy_return_type', result_add_6899)
        
        # ################# End of '__format_admitted_params(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__format_admitted_params' in the type store
        # Getting the type of 'stypy_return_type' (line 299)
        stypy_return_type_6900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6900)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__format_admitted_params'
        return stypy_return_type_6900


    @staticmethod
    @norecursion
    def __compare(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__compare'
        module_type_store = module_type_store.open_function_context('__compare', 353, 4, False)
        
        # Passed parameters checking function
        TypeRuleCallHandler.__compare.__dict__.__setitem__('stypy_localization', localization)
        TypeRuleCallHandler.__compare.__dict__.__setitem__('stypy_type_of_self', None)
        TypeRuleCallHandler.__compare.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeRuleCallHandler.__compare.__dict__.__setitem__('stypy_function_name', '__compare')
        TypeRuleCallHandler.__compare.__dict__.__setitem__('stypy_param_names_list', ['params_in_rules', 'argument_types'])
        TypeRuleCallHandler.__compare.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeRuleCallHandler.__compare.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeRuleCallHandler.__compare.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeRuleCallHandler.__compare.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeRuleCallHandler.__compare.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeRuleCallHandler.__compare.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, None, module_type_store, '__compare', ['params_in_rules', 'argument_types'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__compare', localization, ['argument_types'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__compare(...)' code ##################

        str_6901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, (-1)), 'str', '\n        Most important function in the call handler, determines if a rule matches with the call arguments initially\n        (this means that the rule can potentially match with the argument types because the structure of the arguments,\n        but if the rule has dependent types, this match could not be so in the end, once the dependent types are\n        evaluated.\n        :param params_in_rules: Parameters declared on the rule\n        :param argument_types: Types passed on the call\n        :return:\n        ')
        
        
        # Call to range(...): (line 364)
        # Processing the call arguments (line 364)
        
        # Call to len(...): (line 364)
        # Processing the call arguments (line 364)
        # Getting the type of 'params_in_rules' (line 364)
        params_in_rules_6904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 27), 'params_in_rules', False)
        # Processing the call keyword arguments (line 364)
        kwargs_6905 = {}
        # Getting the type of 'len' (line 364)
        len_6903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 23), 'len', False)
        # Calling len(args, kwargs) (line 364)
        len_call_result_6906 = invoke(stypy.reporting.localization.Localization(__file__, 364, 23), len_6903, *[params_in_rules_6904], **kwargs_6905)
        
        # Processing the call keyword arguments (line 364)
        kwargs_6907 = {}
        # Getting the type of 'range' (line 364)
        range_6902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 17), 'range', False)
        # Calling range(args, kwargs) (line 364)
        range_call_result_6908 = invoke(stypy.reporting.localization.Localization(__file__, 364, 17), range_6902, *[len_call_result_6906], **kwargs_6907)
        
        # Assigning a type to the variable 'range_call_result_6908' (line 364)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 8), 'range_call_result_6908', range_call_result_6908)
        # Testing if the for loop is going to be iterated (line 364)
        # Testing the type of a for loop iterable (line 364)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 364, 8), range_call_result_6908)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 364, 8), range_call_result_6908):
            # Getting the type of the for loop variable (line 364)
            for_loop_var_6909 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 364, 8), range_call_result_6908)
            # Assigning a type to the variable 'i' (line 364)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 8), 'i', for_loop_var_6909)
            # SSA begins for a for statement (line 364)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Subscript to a Name (line 365):
            
            # Assigning a Subscript to a Name (line 365):
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 365)
            i_6910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 36), 'i')
            # Getting the type of 'params_in_rules' (line 365)
            params_in_rules_6911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 20), 'params_in_rules')
            # Obtaining the member '__getitem__' of a type (line 365)
            getitem___6912 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 365, 20), params_in_rules_6911, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 365)
            subscript_call_result_6913 = invoke(stypy.reporting.localization.Localization(__file__, 365, 20), getitem___6912, i_6910)
            
            # Assigning a type to the variable 'param' (line 365)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 12), 'param', subscript_call_result_6913)
            
            # Call to isinstance(...): (line 367)
            # Processing the call arguments (line 367)
            # Getting the type of 'param' (line 367)
            param_6915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 26), 'param', False)
            # Getting the type of 'VarArgType' (line 367)
            VarArgType_6916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 33), 'VarArgType', False)
            # Processing the call keyword arguments (line 367)
            kwargs_6917 = {}
            # Getting the type of 'isinstance' (line 367)
            isinstance_6914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 15), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 367)
            isinstance_call_result_6918 = invoke(stypy.reporting.localization.Localization(__file__, 367, 15), isinstance_6914, *[param_6915, VarArgType_6916], **kwargs_6917)
            
            # Testing if the type of an if condition is none (line 367)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 367, 12), isinstance_call_result_6918):
                pass
            else:
                
                # Testing the type of an if condition (line 367)
                if_condition_6919 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 367, 12), isinstance_call_result_6918)
                # Assigning a type to the variable 'if_condition_6919' (line 367)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 367, 12), 'if_condition_6919', if_condition_6919)
                # SSA begins for if statement (line 367)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # SSA join for if statement (line 367)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Call to isinstance(...): (line 370)
            # Processing the call arguments (line 370)
            # Getting the type of 'param' (line 370)
            param_6921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 26), 'param', False)
            # Getting the type of 'BaseTypeGroup' (line 370)
            BaseTypeGroup_6922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 33), 'BaseTypeGroup', False)
            # Processing the call keyword arguments (line 370)
            kwargs_6923 = {}
            # Getting the type of 'isinstance' (line 370)
            isinstance_6920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 15), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 370)
            isinstance_call_result_6924 = invoke(stypy.reporting.localization.Localization(__file__, 370, 15), isinstance_6920, *[param_6921, BaseTypeGroup_6922], **kwargs_6923)
            
            # Testing if the type of an if condition is none (line 370)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 370, 12), isinstance_call_result_6924):
                
                
                # Getting the type of 'param' (line 375)
                param_6935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 23), 'param')
                
                # Call to get_python_type(...): (line 375)
                # Processing the call keyword arguments (line 375)
                kwargs_6941 = {}
                
                # Obtaining the type of the subscript
                # Getting the type of 'i' (line 375)
                i_6936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 47), 'i', False)
                # Getting the type of 'argument_types' (line 375)
                argument_types_6937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 32), 'argument_types', False)
                # Obtaining the member '__getitem__' of a type (line 375)
                getitem___6938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 375, 32), argument_types_6937, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 375)
                subscript_call_result_6939 = invoke(stypy.reporting.localization.Localization(__file__, 375, 32), getitem___6938, i_6936)
                
                # Obtaining the member 'get_python_type' of a type (line 375)
                get_python_type_6940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 375, 32), subscript_call_result_6939, 'get_python_type')
                # Calling get_python_type(args, kwargs) (line 375)
                get_python_type_call_result_6942 = invoke(stypy.reporting.localization.Localization(__file__, 375, 32), get_python_type_6940, *[], **kwargs_6941)
                
                # Applying the binary operator '==' (line 375)
                result_eq_6943 = python_operator(stypy.reporting.localization.Localization(__file__, 375, 23), '==', param_6935, get_python_type_call_result_6942)
                
                # Applying the 'not' unary operator (line 375)
                result_not__6944 = python_operator(stypy.reporting.localization.Localization(__file__, 375, 19), 'not', result_eq_6943)
                
                # Testing if the type of an if condition is none (line 375)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 375, 16), result_not__6944):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 375)
                    if_condition_6945 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 375, 16), result_not__6944)
                    # Assigning a type to the variable 'if_condition_6945' (line 375)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 16), 'if_condition_6945', if_condition_6945)
                    # SSA begins for if statement (line 375)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    # Getting the type of 'False' (line 376)
                    False_6946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 27), 'False')
                    # Assigning a type to the variable 'stypy_return_type' (line 376)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 20), 'stypy_return_type', False_6946)
                    # SSA join for if statement (line 375)
                    module_type_store = module_type_store.join_ssa_context()
                    

            else:
                
                # Testing the type of an if condition (line 370)
                if_condition_6925 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 370, 12), isinstance_call_result_6924)
                # Assigning a type to the variable 'if_condition_6925' (line 370)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 12), 'if_condition_6925', if_condition_6925)
                # SSA begins for if statement (line 370)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                
                # Getting the type of 'param' (line 371)
                param_6926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 23), 'param')
                
                # Obtaining the type of the subscript
                # Getting the type of 'i' (line 371)
                i_6927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 47), 'i')
                # Getting the type of 'argument_types' (line 371)
                argument_types_6928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 32), 'argument_types')
                # Obtaining the member '__getitem__' of a type (line 371)
                getitem___6929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 32), argument_types_6928, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 371)
                subscript_call_result_6930 = invoke(stypy.reporting.localization.Localization(__file__, 371, 32), getitem___6929, i_6927)
                
                # Applying the binary operator '==' (line 371)
                result_eq_6931 = python_operator(stypy.reporting.localization.Localization(__file__, 371, 23), '==', param_6926, subscript_call_result_6930)
                
                # Applying the 'not' unary operator (line 371)
                result_not__6932 = python_operator(stypy.reporting.localization.Localization(__file__, 371, 19), 'not', result_eq_6931)
                
                # Testing if the type of an if condition is none (line 371)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 371, 16), result_not__6932):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 371)
                    if_condition_6933 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 371, 16), result_not__6932)
                    # Assigning a type to the variable 'if_condition_6933' (line 371)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 16), 'if_condition_6933', if_condition_6933)
                    # SSA begins for if statement (line 371)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    # Getting the type of 'False' (line 372)
                    False_6934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 27), 'False')
                    # Assigning a type to the variable 'stypy_return_type' (line 372)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 20), 'stypy_return_type', False_6934)
                    # SSA join for if statement (line 371)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA branch for the else part of an if statement (line 370)
                module_type_store.open_ssa_branch('else')
                
                
                # Getting the type of 'param' (line 375)
                param_6935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 23), 'param')
                
                # Call to get_python_type(...): (line 375)
                # Processing the call keyword arguments (line 375)
                kwargs_6941 = {}
                
                # Obtaining the type of the subscript
                # Getting the type of 'i' (line 375)
                i_6936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 47), 'i', False)
                # Getting the type of 'argument_types' (line 375)
                argument_types_6937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 32), 'argument_types', False)
                # Obtaining the member '__getitem__' of a type (line 375)
                getitem___6938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 375, 32), argument_types_6937, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 375)
                subscript_call_result_6939 = invoke(stypy.reporting.localization.Localization(__file__, 375, 32), getitem___6938, i_6936)
                
                # Obtaining the member 'get_python_type' of a type (line 375)
                get_python_type_6940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 375, 32), subscript_call_result_6939, 'get_python_type')
                # Calling get_python_type(args, kwargs) (line 375)
                get_python_type_call_result_6942 = invoke(stypy.reporting.localization.Localization(__file__, 375, 32), get_python_type_6940, *[], **kwargs_6941)
                
                # Applying the binary operator '==' (line 375)
                result_eq_6943 = python_operator(stypy.reporting.localization.Localization(__file__, 375, 23), '==', param_6935, get_python_type_call_result_6942)
                
                # Applying the 'not' unary operator (line 375)
                result_not__6944 = python_operator(stypy.reporting.localization.Localization(__file__, 375, 19), 'not', result_eq_6943)
                
                # Testing if the type of an if condition is none (line 375)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 375, 16), result_not__6944):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 375)
                    if_condition_6945 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 375, 16), result_not__6944)
                    # Assigning a type to the variable 'if_condition_6945' (line 375)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 16), 'if_condition_6945', if_condition_6945)
                    # SSA begins for if statement (line 375)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    # Getting the type of 'False' (line 376)
                    False_6946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 27), 'False')
                    # Assigning a type to the variable 'stypy_return_type' (line 376)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 20), 'stypy_return_type', False_6946)
                    # SSA join for if statement (line 375)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for if statement (line 370)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 'True' (line 378)
        True_6947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 15), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 378)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 8), 'stypy_return_type', True_6947)
        
        # ################# End of '__compare(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__compare' in the type store
        # Getting the type of 'stypy_return_type' (line 353)
        stypy_return_type_6948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6948)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__compare'
        return stypy_return_type_6948


    @staticmethod
    @norecursion
    def __create_return_type(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__create_return_type'
        module_type_store = module_type_store.open_function_context('__create_return_type', 380, 4, False)
        
        # Passed parameters checking function
        TypeRuleCallHandler.__create_return_type.__dict__.__setitem__('stypy_localization', localization)
        TypeRuleCallHandler.__create_return_type.__dict__.__setitem__('stypy_type_of_self', None)
        TypeRuleCallHandler.__create_return_type.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeRuleCallHandler.__create_return_type.__dict__.__setitem__('stypy_function_name', '__create_return_type')
        TypeRuleCallHandler.__create_return_type.__dict__.__setitem__('stypy_param_names_list', ['localization', 'ret_type', 'argument_types'])
        TypeRuleCallHandler.__create_return_type.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeRuleCallHandler.__create_return_type.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeRuleCallHandler.__create_return_type.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeRuleCallHandler.__create_return_type.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeRuleCallHandler.__create_return_type.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeRuleCallHandler.__create_return_type.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, None, module_type_store, '__create_return_type', ['localization', 'ret_type', 'argument_types'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__create_return_type', localization, ['ret_type', 'argument_types'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__create_return_type(...)' code ##################

        str_6949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, (-1)), 'str', '\n        Create a suitable return type for the rule (if the return type is a dependent type, this invoked it against\n        the call arguments to obtain it)\n        :param localization: Caller information\n        :param ret_type: Declared return type in a matched rule\n        :param argument_types: Arguments of the call\n        :return:\n        ')
        
        # Call to isinstance(...): (line 390)
        # Processing the call arguments (line 390)
        # Getting the type of 'ret_type' (line 390)
        ret_type_6951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 22), 'ret_type', False)
        # Getting the type of 'DependentType' (line 390)
        DependentType_6952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 32), 'DependentType', False)
        # Processing the call keyword arguments (line 390)
        kwargs_6953 = {}
        # Getting the type of 'isinstance' (line 390)
        isinstance_6950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 390)
        isinstance_call_result_6954 = invoke(stypy.reporting.localization.Localization(__file__, 390, 11), isinstance_6950, *[ret_type_6951, DependentType_6952], **kwargs_6953)
        
        # Testing if the type of an if condition is none (line 390)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 390, 8), isinstance_call_result_6954):
            
            # Call to instance(...): (line 394)
            # Processing the call arguments (line 394)
            # Getting the type of 'ret_type' (line 395)
            ret_type_6971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 16), 'ret_type', False)
            # Processing the call keyword arguments (line 394)
            kwargs_6972 = {}
            # Getting the type of 'type_inference_copy' (line 394)
            type_inference_copy_6967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 19), 'type_inference_copy', False)
            # Obtaining the member 'type_inference_proxy' of a type (line 394)
            type_inference_proxy_6968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 19), type_inference_copy_6967, 'type_inference_proxy')
            # Obtaining the member 'TypeInferenceProxy' of a type (line 394)
            TypeInferenceProxy_6969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 19), type_inference_proxy_6968, 'TypeInferenceProxy')
            # Obtaining the member 'instance' of a type (line 394)
            instance_6970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 19), TypeInferenceProxy_6969, 'instance')
            # Calling instance(args, kwargs) (line 394)
            instance_call_result_6973 = invoke(stypy.reporting.localization.Localization(__file__, 394, 19), instance_6970, *[ret_type_6971], **kwargs_6972)
            
            # Assigning a type to the variable 'stypy_return_type' (line 394)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 394, 12), 'stypy_return_type', instance_call_result_6973)
        else:
            
            # Testing the type of an if condition (line 390)
            if_condition_6955 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 390, 8), isinstance_call_result_6954)
            # Assigning a type to the variable 'if_condition_6955' (line 390)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 8), 'if_condition_6955', if_condition_6955)
            # SSA begins for if statement (line 390)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to instance(...): (line 391)
            # Processing the call arguments (line 391)
            
            # Call to ret_type(...): (line 392)
            # Processing the call arguments (line 392)
            # Getting the type of 'localization' (line 392)
            localization_6961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 25), 'localization', False)
            # Getting the type of 'argument_types' (line 392)
            argument_types_6962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 39), 'argument_types', False)
            # Processing the call keyword arguments (line 392)
            kwargs_6963 = {}
            # Getting the type of 'ret_type' (line 392)
            ret_type_6960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 16), 'ret_type', False)
            # Calling ret_type(args, kwargs) (line 392)
            ret_type_call_result_6964 = invoke(stypy.reporting.localization.Localization(__file__, 392, 16), ret_type_6960, *[localization_6961, argument_types_6962], **kwargs_6963)
            
            # Processing the call keyword arguments (line 391)
            kwargs_6965 = {}
            # Getting the type of 'type_inference_copy' (line 391)
            type_inference_copy_6956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 19), 'type_inference_copy', False)
            # Obtaining the member 'type_inference_proxy' of a type (line 391)
            type_inference_proxy_6957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 19), type_inference_copy_6956, 'type_inference_proxy')
            # Obtaining the member 'TypeInferenceProxy' of a type (line 391)
            TypeInferenceProxy_6958 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 19), type_inference_proxy_6957, 'TypeInferenceProxy')
            # Obtaining the member 'instance' of a type (line 391)
            instance_6959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 19), TypeInferenceProxy_6958, 'instance')
            # Calling instance(args, kwargs) (line 391)
            instance_call_result_6966 = invoke(stypy.reporting.localization.Localization(__file__, 391, 19), instance_6959, *[ret_type_call_result_6964], **kwargs_6965)
            
            # Assigning a type to the variable 'stypy_return_type' (line 391)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 12), 'stypy_return_type', instance_call_result_6966)
            # SSA branch for the else part of an if statement (line 390)
            module_type_store.open_ssa_branch('else')
            
            # Call to instance(...): (line 394)
            # Processing the call arguments (line 394)
            # Getting the type of 'ret_type' (line 395)
            ret_type_6971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 16), 'ret_type', False)
            # Processing the call keyword arguments (line 394)
            kwargs_6972 = {}
            # Getting the type of 'type_inference_copy' (line 394)
            type_inference_copy_6967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 19), 'type_inference_copy', False)
            # Obtaining the member 'type_inference_proxy' of a type (line 394)
            type_inference_proxy_6968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 19), type_inference_copy_6967, 'type_inference_proxy')
            # Obtaining the member 'TypeInferenceProxy' of a type (line 394)
            TypeInferenceProxy_6969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 19), type_inference_proxy_6968, 'TypeInferenceProxy')
            # Obtaining the member 'instance' of a type (line 394)
            instance_6970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 19), TypeInferenceProxy_6969, 'instance')
            # Calling instance(args, kwargs) (line 394)
            instance_call_result_6973 = invoke(stypy.reporting.localization.Localization(__file__, 394, 19), instance_6970, *[ret_type_6971], **kwargs_6972)
            
            # Assigning a type to the variable 'stypy_return_type' (line 394)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 394, 12), 'stypy_return_type', instance_call_result_6973)
            # SSA join for if statement (line 390)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of '__create_return_type(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__create_return_type' in the type store
        # Getting the type of 'stypy_return_type' (line 380)
        stypy_return_type_6974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6974)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__create_return_type'
        return stypy_return_type_6974


    @norecursion
    def get_parameter_arity(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_parameter_arity'
        module_type_store = module_type_store.open_function_context('get_parameter_arity', 397, 4, False)
        # Assigning a type to the variable 'self' (line 398)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeRuleCallHandler.get_parameter_arity.__dict__.__setitem__('stypy_localization', localization)
        TypeRuleCallHandler.get_parameter_arity.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeRuleCallHandler.get_parameter_arity.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeRuleCallHandler.get_parameter_arity.__dict__.__setitem__('stypy_function_name', 'TypeRuleCallHandler.get_parameter_arity')
        TypeRuleCallHandler.get_parameter_arity.__dict__.__setitem__('stypy_param_names_list', ['proxy_obj', 'callable_entity'])
        TypeRuleCallHandler.get_parameter_arity.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeRuleCallHandler.get_parameter_arity.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeRuleCallHandler.get_parameter_arity.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeRuleCallHandler.get_parameter_arity.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeRuleCallHandler.get_parameter_arity.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeRuleCallHandler.get_parameter_arity.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeRuleCallHandler.get_parameter_arity', ['proxy_obj', 'callable_entity'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_parameter_arity', localization, ['proxy_obj', 'callable_entity'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_parameter_arity(...)' code ##################

        str_6975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, (-1)), 'str', '\n        Obtain the minimum and maximum arity of a callable element using the type rules declared for it. It also\n        indicates if it has varargs (infinite arity)\n        :param proxy_obj: TypeInferenceProxy that holds the callable entity\n        :param callable_entity: Callable entity\n        :return: list of possible arities, bool (wether it has varargs or not)\n        ')
        
        # Call to isclass(...): (line 405)
        # Processing the call arguments (line 405)
        # Getting the type of 'callable_entity' (line 405)
        callable_entity_6978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 27), 'callable_entity', False)
        # Processing the call keyword arguments (line 405)
        kwargs_6979 = {}
        # Getting the type of 'inspect' (line 405)
        inspect_6976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 11), 'inspect', False)
        # Obtaining the member 'isclass' of a type (line 405)
        isclass_6977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 11), inspect_6976, 'isclass')
        # Calling isclass(args, kwargs) (line 405)
        isclass_call_result_6980 = invoke(stypy.reporting.localization.Localization(__file__, 405, 11), isclass_6977, *[callable_entity_6978], **kwargs_6979)
        
        # Testing if the type of an if condition is none (line 405)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 405, 8), isclass_call_result_6980):
            
            # Assigning a Attribute to a Name (line 408):
            
            # Assigning a Attribute to a Name (line 408):
            # Getting the type of 'proxy_obj' (line 408)
            proxy_obj_6986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 25), 'proxy_obj')
            # Obtaining the member 'name' of a type (line 408)
            name_6987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 25), proxy_obj_6986, 'name')
            # Assigning a type to the variable 'cache_name' (line 408)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 408, 12), 'cache_name', name_6987)
        else:
            
            # Testing the type of an if condition (line 405)
            if_condition_6981 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 405, 8), isclass_call_result_6980)
            # Assigning a type to the variable 'if_condition_6981' (line 405)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 8), 'if_condition_6981', if_condition_6981)
            # SSA begins for if statement (line 405)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a BinOp to a Name (line 406):
            
            # Assigning a BinOp to a Name (line 406):
            # Getting the type of 'proxy_obj' (line 406)
            proxy_obj_6982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 25), 'proxy_obj')
            # Obtaining the member 'name' of a type (line 406)
            name_6983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 25), proxy_obj_6982, 'name')
            str_6984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 42), 'str', '.__init__')
            # Applying the binary operator '+' (line 406)
            result_add_6985 = python_operator(stypy.reporting.localization.Localization(__file__, 406, 25), '+', name_6983, str_6984)
            
            # Assigning a type to the variable 'cache_name' (line 406)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 12), 'cache_name', result_add_6985)
            # SSA branch for the else part of an if statement (line 405)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Attribute to a Name (line 408):
            
            # Assigning a Attribute to a Name (line 408):
            # Getting the type of 'proxy_obj' (line 408)
            proxy_obj_6986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 25), 'proxy_obj')
            # Obtaining the member 'name' of a type (line 408)
            name_6987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 25), proxy_obj_6986, 'name')
            # Assigning a type to the variable 'cache_name' (line 408)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 408, 12), 'cache_name', name_6987)
            # SSA join for if statement (line 405)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Name to a Name (line 410):
        
        # Assigning a Name to a Name (line 410):
        # Getting the type of 'False' (line 410)
        False_6988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 22), 'False')
        # Assigning a type to the variable 'has_varargs' (line 410)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 410, 8), 'has_varargs', False_6988)
        
        # Assigning a List to a Name (line 411):
        
        # Assigning a List to a Name (line 411):
        
        # Obtaining an instance of the builtin type 'list' (line 411)
        list_6989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 411)
        
        # Assigning a type to the variable 'arities' (line 411)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 8), 'arities', list_6989)
        
        # Assigning a Call to a Tuple (line 412):
        
        # Assigning a Call to a Name:
        
        # Call to __get_rules_and_name(...): (line 412)
        # Processing the call arguments (line 412)
        # Getting the type of 'cache_name' (line 412)
        cache_name_6992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 48), 'cache_name', False)
        # Getting the type of 'proxy_obj' (line 412)
        proxy_obj_6993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 60), 'proxy_obj', False)
        # Obtaining the member 'parent_proxy' of a type (line 412)
        parent_proxy_6994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 60), proxy_obj_6993, 'parent_proxy')
        # Obtaining the member 'name' of a type (line 412)
        name_6995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 60), parent_proxy_6994, 'name')
        # Processing the call keyword arguments (line 412)
        kwargs_6996 = {}
        # Getting the type of 'self' (line 412)
        self_6990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 22), 'self', False)
        # Obtaining the member '__get_rules_and_name' of a type (line 412)
        get_rules_and_name_6991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 22), self_6990, '__get_rules_and_name')
        # Calling __get_rules_and_name(args, kwargs) (line 412)
        get_rules_and_name_call_result_6997 = invoke(stypy.reporting.localization.Localization(__file__, 412, 22), get_rules_and_name_6991, *[cache_name_6992, name_6995], **kwargs_6996)
        
        # Assigning a type to the variable 'call_assignment_6187' (line 412)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 412, 8), 'call_assignment_6187', get_rules_and_name_call_result_6997)
        
        # Assigning a Call to a Name (line 412):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_6187' (line 412)
        call_assignment_6187_6998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 8), 'call_assignment_6187', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_6999 = stypy_get_value_from_tuple(call_assignment_6187_6998, 2, 0)
        
        # Assigning a type to the variable 'call_assignment_6188' (line 412)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 412, 8), 'call_assignment_6188', stypy_get_value_from_tuple_call_result_6999)
        
        # Assigning a Name to a Name (line 412):
        # Getting the type of 'call_assignment_6188' (line 412)
        call_assignment_6188_7000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 8), 'call_assignment_6188')
        # Assigning a type to the variable 'name' (line 412)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 412, 8), 'name', call_assignment_6188_7000)
        
        # Assigning a Call to a Name (line 412):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_6187' (line 412)
        call_assignment_6187_7001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 8), 'call_assignment_6187', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_7002 = stypy_get_value_from_tuple(call_assignment_6187_7001, 2, 1)
        
        # Assigning a type to the variable 'call_assignment_6189' (line 412)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 412, 8), 'call_assignment_6189', stypy_get_value_from_tuple_call_result_7002)
        
        # Assigning a Name to a Name (line 412):
        # Getting the type of 'call_assignment_6189' (line 412)
        call_assignment_6189_7003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 8), 'call_assignment_6189')
        # Assigning a type to the variable 'rules' (line 412)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 412, 14), 'rules', call_assignment_6189_7003)
        
        # Getting the type of 'rules' (line 413)
        rules_7004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 46), 'rules')
        # Assigning a type to the variable 'rules_7004' (line 413)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 413, 8), 'rules_7004', rules_7004)
        # Testing if the for loop is going to be iterated (line 413)
        # Testing the type of a for loop iterable (line 413)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 413, 8), rules_7004)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 413, 8), rules_7004):
            # Getting the type of the for loop variable (line 413)
            for_loop_var_7005 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 413, 8), rules_7004)
            # Assigning a type to the variable 'params_in_rules' (line 413)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 413, 8), 'params_in_rules', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 413, 8), for_loop_var_7005, 2, 0))
            # Assigning a type to the variable 'return_type' (line 413)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 413, 8), 'return_type', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 413, 8), for_loop_var_7005, 2, 1))
            # SSA begins for a for statement (line 413)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to __has_varargs_in_rule_params(...): (line 414)
            # Processing the call arguments (line 414)
            # Getting the type of 'params_in_rules' (line 414)
            params_in_rules_7008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 49), 'params_in_rules', False)
            # Processing the call keyword arguments (line 414)
            kwargs_7009 = {}
            # Getting the type of 'self' (line 414)
            self_7006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 15), 'self', False)
            # Obtaining the member '__has_varargs_in_rule_params' of a type (line 414)
            has_varargs_in_rule_params_7007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 414, 15), self_7006, '__has_varargs_in_rule_params')
            # Calling __has_varargs_in_rule_params(args, kwargs) (line 414)
            has_varargs_in_rule_params_call_result_7010 = invoke(stypy.reporting.localization.Localization(__file__, 414, 15), has_varargs_in_rule_params_7007, *[params_in_rules_7008], **kwargs_7009)
            
            # Testing if the type of an if condition is none (line 414)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 414, 12), has_varargs_in_rule_params_call_result_7010):
                pass
            else:
                
                # Testing the type of an if condition (line 414)
                if_condition_7011 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 414, 12), has_varargs_in_rule_params_call_result_7010)
                # Assigning a type to the variable 'if_condition_7011' (line 414)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 414, 12), 'if_condition_7011', if_condition_7011)
                # SSA begins for if statement (line 414)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Name to a Name (line 415):
                
                # Assigning a Name to a Name (line 415):
                # Getting the type of 'True' (line 415)
                True_7012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 30), 'True')
                # Assigning a type to the variable 'has_varargs' (line 415)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 415, 16), 'has_varargs', True_7012)
                # SSA join for if statement (line 414)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Assigning a Call to a Name (line 416):
            
            # Assigning a Call to a Name (line 416):
            
            # Call to len(...): (line 416)
            # Processing the call arguments (line 416)
            # Getting the type of 'params_in_rules' (line 416)
            params_in_rules_7014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 22), 'params_in_rules', False)
            # Processing the call keyword arguments (line 416)
            kwargs_7015 = {}
            # Getting the type of 'len' (line 416)
            len_7013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 18), 'len', False)
            # Calling len(args, kwargs) (line 416)
            len_call_result_7016 = invoke(stypy.reporting.localization.Localization(__file__, 416, 18), len_7013, *[params_in_rules_7014], **kwargs_7015)
            
            # Assigning a type to the variable 'num' (line 416)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 416, 12), 'num', len_call_result_7016)
            
            # Getting the type of 'num' (line 417)
            num_7017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 15), 'num')
            # Getting the type of 'arities' (line 417)
            arities_7018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 26), 'arities')
            # Applying the binary operator 'notin' (line 417)
            result_contains_7019 = python_operator(stypy.reporting.localization.Localization(__file__, 417, 15), 'notin', num_7017, arities_7018)
            
            # Testing if the type of an if condition is none (line 417)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 417, 12), result_contains_7019):
                pass
            else:
                
                # Testing the type of an if condition (line 417)
                if_condition_7020 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 417, 12), result_contains_7019)
                # Assigning a type to the variable 'if_condition_7020' (line 417)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 417, 12), 'if_condition_7020', if_condition_7020)
                # SSA begins for if statement (line 417)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to append(...): (line 418)
                # Processing the call arguments (line 418)
                # Getting the type of 'num' (line 418)
                num_7023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 31), 'num', False)
                # Processing the call keyword arguments (line 418)
                kwargs_7024 = {}
                # Getting the type of 'arities' (line 418)
                arities_7021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 16), 'arities', False)
                # Obtaining the member 'append' of a type (line 418)
                append_7022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 418, 16), arities_7021, 'append')
                # Calling append(args, kwargs) (line 418)
                append_call_result_7025 = invoke(stypy.reporting.localization.Localization(__file__, 418, 16), append_7022, *[num_7023], **kwargs_7024)
                
                # SSA join for if statement (line 417)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Obtaining an instance of the builtin type 'tuple' (line 420)
        tuple_7026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 420, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 420)
        # Adding element type (line 420)
        # Getting the type of 'arities' (line 420)
        arities_7027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 15), 'arities')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 420, 15), tuple_7026, arities_7027)
        # Adding element type (line 420)
        # Getting the type of 'has_varargs' (line 420)
        has_varargs_7028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 24), 'has_varargs')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 420, 15), tuple_7026, has_varargs_7028)
        
        # Assigning a type to the variable 'stypy_return_type' (line 420)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 420, 8), 'stypy_return_type', tuple_7026)
        
        # ################# End of 'get_parameter_arity(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_parameter_arity' in the type store
        # Getting the type of 'stypy_return_type' (line 397)
        stypy_return_type_7029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_7029)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_parameter_arity'
        return stypy_return_type_7029


    @norecursion
    def __call__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__call__'
        module_type_store = module_type_store.open_function_context('__call__', 422, 4, False)
        # Assigning a type to the variable 'self' (line 423)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeRuleCallHandler.__call__.__dict__.__setitem__('stypy_localization', localization)
        TypeRuleCallHandler.__call__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeRuleCallHandler.__call__.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeRuleCallHandler.__call__.__dict__.__setitem__('stypy_function_name', 'TypeRuleCallHandler.__call__')
        TypeRuleCallHandler.__call__.__dict__.__setitem__('stypy_param_names_list', ['proxy_obj', 'localization', 'callable_entity'])
        TypeRuleCallHandler.__call__.__dict__.__setitem__('stypy_varargs_param_name', 'arg_types')
        TypeRuleCallHandler.__call__.__dict__.__setitem__('stypy_kwargs_param_name', 'kwargs_types')
        TypeRuleCallHandler.__call__.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeRuleCallHandler.__call__.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeRuleCallHandler.__call__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeRuleCallHandler.__call__.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeRuleCallHandler.__call__', ['proxy_obj', 'localization', 'callable_entity'], 'arg_types', 'kwargs_types', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__call__', localization, ['proxy_obj', 'localization', 'callable_entity'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__call__(...)' code ##################

        str_7030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, (-1)), 'str', '\n        Calls the callable entity with its type rules to determine its return type.\n\n        :param proxy_obj: TypeInferenceProxy that hold the callable entity\n        :param localization: Caller information\n        :param callable_entity: Callable entity\n        :param arg_types: Arguments\n        :param kwargs_types: Keyword arguments\n        :return: Return type of the call\n        ')
        
        # Call to isclass(...): (line 433)
        # Processing the call arguments (line 433)
        # Getting the type of 'callable_entity' (line 433)
        callable_entity_7033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 27), 'callable_entity', False)
        # Processing the call keyword arguments (line 433)
        kwargs_7034 = {}
        # Getting the type of 'inspect' (line 433)
        inspect_7031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 11), 'inspect', False)
        # Obtaining the member 'isclass' of a type (line 433)
        isclass_7032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 433, 11), inspect_7031, 'isclass')
        # Calling isclass(args, kwargs) (line 433)
        isclass_call_result_7035 = invoke(stypy.reporting.localization.Localization(__file__, 433, 11), isclass_7032, *[callable_entity_7033], **kwargs_7034)
        
        # Testing if the type of an if condition is none (line 433)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 433, 8), isclass_call_result_7035):
            
            # Assigning a Attribute to a Name (line 436):
            
            # Assigning a Attribute to a Name (line 436):
            # Getting the type of 'proxy_obj' (line 436)
            proxy_obj_7041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 25), 'proxy_obj')
            # Obtaining the member 'name' of a type (line 436)
            name_7042 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 25), proxy_obj_7041, 'name')
            # Assigning a type to the variable 'cache_name' (line 436)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 12), 'cache_name', name_7042)
        else:
            
            # Testing the type of an if condition (line 433)
            if_condition_7036 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 433, 8), isclass_call_result_7035)
            # Assigning a type to the variable 'if_condition_7036' (line 433)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 433, 8), 'if_condition_7036', if_condition_7036)
            # SSA begins for if statement (line 433)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a BinOp to a Name (line 434):
            
            # Assigning a BinOp to a Name (line 434):
            # Getting the type of 'proxy_obj' (line 434)
            proxy_obj_7037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 25), 'proxy_obj')
            # Obtaining the member 'name' of a type (line 434)
            name_7038 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 25), proxy_obj_7037, 'name')
            str_7039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 42), 'str', '.__init__')
            # Applying the binary operator '+' (line 434)
            result_add_7040 = python_operator(stypy.reporting.localization.Localization(__file__, 434, 25), '+', name_7038, str_7039)
            
            # Assigning a type to the variable 'cache_name' (line 434)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 12), 'cache_name', result_add_7040)
            # SSA branch for the else part of an if statement (line 433)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Attribute to a Name (line 436):
            
            # Assigning a Attribute to a Name (line 436):
            # Getting the type of 'proxy_obj' (line 436)
            proxy_obj_7041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 25), 'proxy_obj')
            # Obtaining the member 'name' of a type (line 436)
            name_7042 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 25), proxy_obj_7041, 'name')
            # Assigning a type to the variable 'cache_name' (line 436)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 12), 'cache_name', name_7042)
            # SSA join for if statement (line 433)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Call to a Tuple (line 438):
        
        # Assigning a Call to a Name:
        
        # Call to __get_rules_and_name(...): (line 438)
        # Processing the call arguments (line 438)
        # Getting the type of 'cache_name' (line 438)
        cache_name_7045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 48), 'cache_name', False)
        # Getting the type of 'proxy_obj' (line 438)
        proxy_obj_7046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 60), 'proxy_obj', False)
        # Obtaining the member 'parent_proxy' of a type (line 438)
        parent_proxy_7047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 60), proxy_obj_7046, 'parent_proxy')
        # Obtaining the member 'name' of a type (line 438)
        name_7048 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 60), parent_proxy_7047, 'name')
        # Processing the call keyword arguments (line 438)
        kwargs_7049 = {}
        # Getting the type of 'self' (line 438)
        self_7043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 22), 'self', False)
        # Obtaining the member '__get_rules_and_name' of a type (line 438)
        get_rules_and_name_7044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 22), self_7043, '__get_rules_and_name')
        # Calling __get_rules_and_name(args, kwargs) (line 438)
        get_rules_and_name_call_result_7050 = invoke(stypy.reporting.localization.Localization(__file__, 438, 22), get_rules_and_name_7044, *[cache_name_7045, name_7048], **kwargs_7049)
        
        # Assigning a type to the variable 'call_assignment_6190' (line 438)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 8), 'call_assignment_6190', get_rules_and_name_call_result_7050)
        
        # Assigning a Call to a Name (line 438):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_6190' (line 438)
        call_assignment_6190_7051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 8), 'call_assignment_6190', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_7052 = stypy_get_value_from_tuple(call_assignment_6190_7051, 2, 0)
        
        # Assigning a type to the variable 'call_assignment_6191' (line 438)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 8), 'call_assignment_6191', stypy_get_value_from_tuple_call_result_7052)
        
        # Assigning a Name to a Name (line 438):
        # Getting the type of 'call_assignment_6191' (line 438)
        call_assignment_6191_7053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 8), 'call_assignment_6191')
        # Assigning a type to the variable 'name' (line 438)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 8), 'name', call_assignment_6191_7053)
        
        # Assigning a Call to a Name (line 438):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_6190' (line 438)
        call_assignment_6190_7054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 8), 'call_assignment_6190', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_7055 = stypy_get_value_from_tuple(call_assignment_6190_7054, 2, 1)
        
        # Assigning a type to the variable 'call_assignment_6192' (line 438)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 8), 'call_assignment_6192', stypy_get_value_from_tuple_call_result_7055)
        
        # Assigning a Name to a Name (line 438):
        # Getting the type of 'call_assignment_6192' (line 438)
        call_assignment_6192_7056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 8), 'call_assignment_6192')
        # Assigning a type to the variable 'rules' (line 438)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 14), 'rules', call_assignment_6192_7056)
        
        # Assigning a Name to a Name (line 440):
        
        # Assigning a Name to a Name (line 440):
        # Getting the type of 'None' (line 440)
        None_7057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 25), 'None')
        # Assigning a type to the variable 'argument_types' (line 440)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 8), 'argument_types', None_7057)
        
        
        # Call to len(...): (line 443)
        # Processing the call arguments (line 443)
        # Getting the type of 'rules' (line 443)
        rules_7059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 15), 'rules', False)
        # Processing the call keyword arguments (line 443)
        kwargs_7060 = {}
        # Getting the type of 'len' (line 443)
        len_7058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 11), 'len', False)
        # Calling len(args, kwargs) (line 443)
        len_call_result_7061 = invoke(stypy.reporting.localization.Localization(__file__, 443, 11), len_7058, *[rules_7059], **kwargs_7060)
        
        int_7062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 24), 'int')
        # Applying the binary operator '>' (line 443)
        result_gt_7063 = python_operator(stypy.reporting.localization.Localization(__file__, 443, 11), '>', len_call_result_7061, int_7062)
        
        # Testing if the type of an if condition is none (line 443)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 443, 8), result_gt_7063):
            
            # Assigning a Name to a Name (line 446):
            
            # Assigning a Name to a Name (line 446):
            # Getting the type of 'False' (line 446)
            False_7066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 25), 'False')
            # Assigning a type to the variable 'prints_msg' (line 446)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 446, 12), 'prints_msg', False_7066)
        else:
            
            # Testing the type of an if condition (line 443)
            if_condition_7064 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 443, 8), result_gt_7063)
            # Assigning a type to the variable 'if_condition_7064' (line 443)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 443, 8), 'if_condition_7064', if_condition_7064)
            # SSA begins for if statement (line 443)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Name to a Name (line 444):
            
            # Assigning a Name to a Name (line 444):
            # Getting the type of 'True' (line 444)
            True_7065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 25), 'True')
            # Assigning a type to the variable 'prints_msg' (line 444)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 12), 'prints_msg', True_7065)
            # SSA branch for the else part of an if statement (line 443)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Name to a Name (line 446):
            
            # Assigning a Name to a Name (line 446):
            # Getting the type of 'False' (line 446)
            False_7066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 25), 'False')
            # Assigning a type to the variable 'prints_msg' (line 446)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 446, 12), 'prints_msg', False_7066)
            # SSA join for if statement (line 443)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Evaluating a boolean operation
        
        # Call to ismethod(...): (line 449)
        # Processing the call arguments (line 449)
        # Getting the type of 'callable_entity' (line 449)
        callable_entity_7069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 28), 'callable_entity', False)
        # Processing the call keyword arguments (line 449)
        kwargs_7070 = {}
        # Getting the type of 'inspect' (line 449)
        inspect_7067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 11), 'inspect', False)
        # Obtaining the member 'ismethod' of a type (line 449)
        ismethod_7068 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 11), inspect_7067, 'ismethod')
        # Calling ismethod(args, kwargs) (line 449)
        ismethod_call_result_7071 = invoke(stypy.reporting.localization.Localization(__file__, 449, 11), ismethod_7068, *[callable_entity_7069], **kwargs_7070)
        
        
        # Call to ismethoddescriptor(...): (line 449)
        # Processing the call arguments (line 449)
        # Getting the type of 'callable_entity' (line 449)
        callable_entity_7074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 75), 'callable_entity', False)
        # Processing the call keyword arguments (line 449)
        kwargs_7075 = {}
        # Getting the type of 'inspect' (line 449)
        inspect_7072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 48), 'inspect', False)
        # Obtaining the member 'ismethoddescriptor' of a type (line 449)
        ismethoddescriptor_7073 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 48), inspect_7072, 'ismethoddescriptor')
        # Calling ismethoddescriptor(args, kwargs) (line 449)
        ismethoddescriptor_call_result_7076 = invoke(stypy.reporting.localization.Localization(__file__, 449, 48), ismethoddescriptor_7073, *[callable_entity_7074], **kwargs_7075)
        
        # Applying the binary operator 'or' (line 449)
        result_or_keyword_7077 = python_operator(stypy.reporting.localization.Localization(__file__, 449, 11), 'or', ismethod_call_result_7071, ismethoddescriptor_call_result_7076)
        
        # Testing if the type of an if condition is none (line 449)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 449, 8), result_or_keyword_7077):
            pass
        else:
            
            # Testing the type of an if condition (line 449)
            if_condition_7078 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 449, 8), result_or_keyword_7077)
            # Assigning a type to the variable 'if_condition_7078' (line 449)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 449, 8), 'if_condition_7078', if_condition_7078)
            # SSA begins for if statement (line 449)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            
            # Call to is_type_instance(...): (line 451)
            # Processing the call keyword arguments (line 451)
            kwargs_7082 = {}
            # Getting the type of 'proxy_obj' (line 451)
            proxy_obj_7079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 19), 'proxy_obj', False)
            # Obtaining the member 'parent_proxy' of a type (line 451)
            parent_proxy_7080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 451, 19), proxy_obj_7079, 'parent_proxy')
            # Obtaining the member 'is_type_instance' of a type (line 451)
            is_type_instance_7081 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 451, 19), parent_proxy_7080, 'is_type_instance')
            # Calling is_type_instance(args, kwargs) (line 451)
            is_type_instance_call_result_7083 = invoke(stypy.reporting.localization.Localization(__file__, 451, 19), is_type_instance_7081, *[], **kwargs_7082)
            
            # Applying the 'not' unary operator (line 451)
            result_not__7084 = python_operator(stypy.reporting.localization.Localization(__file__, 451, 15), 'not', is_type_instance_call_result_7083)
            
            # Testing if the type of an if condition is none (line 451)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 451, 12), result_not__7084):
                pass
            else:
                
                # Testing the type of an if condition (line 451)
                if_condition_7085 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 451, 12), result_not__7084)
                # Assigning a type to the variable 'if_condition_7085' (line 451)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 451, 12), 'if_condition_7085', if_condition_7085)
                # SSA begins for if statement (line 451)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                
                # Call to issubclass(...): (line 453)
                # Processing the call arguments (line 453)
                
                # Obtaining the type of the subscript
                int_7087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 453, 44), 'int')
                # Getting the type of 'arg_types' (line 453)
                arg_types_7088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 34), 'arg_types', False)
                # Obtaining the member '__getitem__' of a type (line 453)
                getitem___7089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 453, 34), arg_types_7088, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 453)
                subscript_call_result_7090 = invoke(stypy.reporting.localization.Localization(__file__, 453, 34), getitem___7089, int_7087)
                
                # Obtaining the member 'python_entity' of a type (line 453)
                python_entity_7091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 453, 34), subscript_call_result_7090, 'python_entity')
                # Getting the type of 'callable_entity' (line 453)
                callable_entity_7092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 62), 'callable_entity', False)
                # Obtaining the member '__objclass__' of a type (line 453)
                objclass___7093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 453, 62), callable_entity_7092, '__objclass__')
                # Processing the call keyword arguments (line 453)
                kwargs_7094 = {}
                # Getting the type of 'issubclass' (line 453)
                issubclass_7086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 23), 'issubclass', False)
                # Calling issubclass(args, kwargs) (line 453)
                issubclass_call_result_7095 = invoke(stypy.reporting.localization.Localization(__file__, 453, 23), issubclass_7086, *[python_entity_7091, objclass___7093], **kwargs_7094)
                
                # Applying the 'not' unary operator (line 453)
                result_not__7096 = python_operator(stypy.reporting.localization.Localization(__file__, 453, 19), 'not', issubclass_call_result_7095)
                
                # Testing if the type of an if condition is none (line 453)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 453, 16), result_not__7096):
                    
                    # Assigning a Call to a Name (line 466):
                    
                    # Assigning a Call to a Name (line 466):
                    
                    # Call to tuple(...): (line 466)
                    # Processing the call arguments (line 466)
                    
                    # Call to list(...): (line 466)
                    # Processing the call arguments (line 466)
                    
                    # Obtaining the type of the subscript
                    int_7159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 466, 58), 'int')
                    slice_7160 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 466, 48), int_7159, None, None)
                    # Getting the type of 'arg_types' (line 466)
                    arg_types_7161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 48), 'arg_types', False)
                    # Obtaining the member '__getitem__' of a type (line 466)
                    getitem___7162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 48), arg_types_7161, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 466)
                    subscript_call_result_7163 = invoke(stypy.reporting.localization.Localization(__file__, 466, 48), getitem___7162, slice_7160)
                    
                    # Processing the call keyword arguments (line 466)
                    kwargs_7164 = {}
                    # Getting the type of 'list' (line 466)
                    list_7158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 43), 'list', False)
                    # Calling list(args, kwargs) (line 466)
                    list_call_result_7165 = invoke(stypy.reporting.localization.Localization(__file__, 466, 43), list_7158, *[subscript_call_result_7163], **kwargs_7164)
                    
                    
                    # Call to values(...): (line 466)
                    # Processing the call keyword arguments (line 466)
                    kwargs_7168 = {}
                    # Getting the type of 'kwargs_types' (line 466)
                    kwargs_types_7166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 65), 'kwargs_types', False)
                    # Obtaining the member 'values' of a type (line 466)
                    values_7167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 65), kwargs_types_7166, 'values')
                    # Calling values(args, kwargs) (line 466)
                    values_call_result_7169 = invoke(stypy.reporting.localization.Localization(__file__, 466, 65), values_7167, *[], **kwargs_7168)
                    
                    # Applying the binary operator '+' (line 466)
                    result_add_7170 = python_operator(stypy.reporting.localization.Localization(__file__, 466, 43), '+', list_call_result_7165, values_call_result_7169)
                    
                    # Processing the call keyword arguments (line 466)
                    kwargs_7171 = {}
                    # Getting the type of 'tuple' (line 466)
                    tuple_7157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 37), 'tuple', False)
                    # Calling tuple(args, kwargs) (line 466)
                    tuple_call_result_7172 = invoke(stypy.reporting.localization.Localization(__file__, 466, 37), tuple_7157, *[result_add_7170], **kwargs_7171)
                    
                    # Assigning a type to the variable 'argument_types' (line 466)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 466, 20), 'argument_types', tuple_call_result_7172)
                else:
                    
                    # Testing the type of an if condition (line 453)
                    if_condition_7097 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 453, 16), result_not__7096)
                    # Assigning a type to the variable 'if_condition_7097' (line 453)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 453, 16), 'if_condition_7097', if_condition_7097)
                    # SSA begins for if statement (line 453)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Call to a Name (line 455):
                    
                    # Assigning a Call to a Name (line 455):
                    
                    # Call to tuple(...): (line 455)
                    # Processing the call arguments (line 455)
                    
                    # Call to list(...): (line 455)
                    # Processing the call arguments (line 455)
                    # Getting the type of 'arg_types' (line 455)
                    arg_types_7100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 48), 'arg_types', False)
                    # Processing the call keyword arguments (line 455)
                    kwargs_7101 = {}
                    # Getting the type of 'list' (line 455)
                    list_7099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 43), 'list', False)
                    # Calling list(args, kwargs) (line 455)
                    list_call_result_7102 = invoke(stypy.reporting.localization.Localization(__file__, 455, 43), list_7099, *[arg_types_7100], **kwargs_7101)
                    
                    
                    # Call to values(...): (line 455)
                    # Processing the call keyword arguments (line 455)
                    kwargs_7105 = {}
                    # Getting the type of 'kwargs_types' (line 455)
                    kwargs_types_7103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 61), 'kwargs_types', False)
                    # Obtaining the member 'values' of a type (line 455)
                    values_7104 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 61), kwargs_types_7103, 'values')
                    # Calling values(args, kwargs) (line 455)
                    values_call_result_7106 = invoke(stypy.reporting.localization.Localization(__file__, 455, 61), values_7104, *[], **kwargs_7105)
                    
                    # Applying the binary operator '+' (line 455)
                    result_add_7107 = python_operator(stypy.reporting.localization.Localization(__file__, 455, 43), '+', list_call_result_7102, values_call_result_7106)
                    
                    # Processing the call keyword arguments (line 455)
                    kwargs_7108 = {}
                    # Getting the type of 'tuple' (line 455)
                    tuple_7098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 37), 'tuple', False)
                    # Calling tuple(args, kwargs) (line 455)
                    tuple_call_result_7109 = invoke(stypy.reporting.localization.Localization(__file__, 455, 37), tuple_7098, *[result_add_7107], **kwargs_7108)
                    
                    # Assigning a type to the variable 'argument_types' (line 455)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 455, 20), 'argument_types', tuple_call_result_7109)
                    
                    # Assigning a Call to a Name (line 456):
                    
                    # Assigning a Call to a Name (line 456):
                    
                    # Call to __format_admitted_params(...): (line 456)
                    # Processing the call arguments (line 456)
                    # Getting the type of 'name' (line 456)
                    name_7112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 63), 'name', False)
                    # Getting the type of 'rules' (line 456)
                    rules_7113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 69), 'rules', False)
                    # Getting the type of 'argument_types' (line 456)
                    argument_types_7114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 76), 'argument_types', False)
                    
                    # Call to len(...): (line 456)
                    # Processing the call arguments (line 456)
                    # Getting the type of 'argument_types' (line 456)
                    argument_types_7116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 96), 'argument_types', False)
                    # Processing the call keyword arguments (line 456)
                    kwargs_7117 = {}
                    # Getting the type of 'len' (line 456)
                    len_7115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 92), 'len', False)
                    # Calling len(args, kwargs) (line 456)
                    len_call_result_7118 = invoke(stypy.reporting.localization.Localization(__file__, 456, 92), len_7115, *[argument_types_7116], **kwargs_7117)
                    
                    # Processing the call keyword arguments (line 456)
                    kwargs_7119 = {}
                    # Getting the type of 'self' (line 456)
                    self_7110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 33), 'self', False)
                    # Obtaining the member '__format_admitted_params' of a type (line 456)
                    format_admitted_params_7111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 33), self_7110, '__format_admitted_params')
                    # Calling __format_admitted_params(args, kwargs) (line 456)
                    format_admitted_params_call_result_7120 = invoke(stypy.reporting.localization.Localization(__file__, 456, 33), format_admitted_params_7111, *[name_7112, rules_7113, argument_types_7114, len_call_result_7118], **kwargs_7119)
                    
                    # Assigning a type to the variable 'usage_hint' (line 456)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 20), 'usage_hint', format_admitted_params_call_result_7120)
                    
                    # Assigning a Call to a Name (line 457):
                    
                    # Assigning a Call to a Name (line 457):
                    
                    # Call to str(...): (line 457)
                    # Processing the call arguments (line 457)
                    # Getting the type of 'argument_types' (line 457)
                    argument_types_7122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 42), 'argument_types', False)
                    # Processing the call keyword arguments (line 457)
                    kwargs_7123 = {}
                    # Getting the type of 'str' (line 457)
                    str_7121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 38), 'str', False)
                    # Calling str(args, kwargs) (line 457)
                    str_call_result_7124 = invoke(stypy.reporting.localization.Localization(__file__, 457, 38), str_7121, *[argument_types_7122], **kwargs_7123)
                    
                    # Assigning a type to the variable 'arg_description' (line 457)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 457, 20), 'arg_description', str_call_result_7124)
                    
                    # Assigning a Call to a Name (line 458):
                    
                    # Assigning a Call to a Name (line 458):
                    
                    # Call to replace(...): (line 458)
                    # Processing the call arguments (line 458)
                    str_7127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 458, 62), 'str', ',)')
                    str_7128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 458, 68), 'str', ')')
                    # Processing the call keyword arguments (line 458)
                    kwargs_7129 = {}
                    # Getting the type of 'arg_description' (line 458)
                    arg_description_7125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 38), 'arg_description', False)
                    # Obtaining the member 'replace' of a type (line 458)
                    replace_7126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 458, 38), arg_description_7125, 'replace')
                    # Calling replace(args, kwargs) (line 458)
                    replace_call_result_7130 = invoke(stypy.reporting.localization.Localization(__file__, 458, 38), replace_7126, *[str_7127, str_7128], **kwargs_7129)
                    
                    # Assigning a type to the variable 'arg_description' (line 458)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 458, 20), 'arg_description', replace_call_result_7130)
                    
                    # Call to TypeError(...): (line 459)
                    # Processing the call arguments (line 459)
                    # Getting the type of 'localization' (line 459)
                    localization_7132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 37), 'localization', False)
                    
                    # Call to format(...): (line 460)
                    # Processing the call arguments (line 460)
                    # Getting the type of 'name' (line 461)
                    name_7135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 62), 'name', False)
                    # Getting the type of 'arg_description' (line 461)
                    arg_description_7136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 68), 'arg_description', False)
                    # Getting the type of 'usage_hint' (line 461)
                    usage_hint_7137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 85), 'usage_hint', False)
                    
                    # Call to str(...): (line 462)
                    # Processing the call arguments (line 462)
                    # Getting the type of 'callable_entity' (line 462)
                    callable_entity_7139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 66), 'callable_entity', False)
                    # Obtaining the member '__objclass__' of a type (line 462)
                    objclass___7140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 462, 66), callable_entity_7139, '__objclass__')
                    # Processing the call keyword arguments (line 462)
                    kwargs_7141 = {}
                    # Getting the type of 'str' (line 462)
                    str_7138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 62), 'str', False)
                    # Calling str(args, kwargs) (line 462)
                    str_call_result_7142 = invoke(stypy.reporting.localization.Localization(__file__, 462, 62), str_7138, *[objclass___7140], **kwargs_7141)
                    
                    
                    # Call to str(...): (line 463)
                    # Processing the call arguments (line 463)
                    
                    # Obtaining the type of the subscript
                    int_7144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 76), 'int')
                    # Getting the type of 'arg_types' (line 463)
                    arg_types_7145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 66), 'arg_types', False)
                    # Obtaining the member '__getitem__' of a type (line 463)
                    getitem___7146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 66), arg_types_7145, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 463)
                    subscript_call_result_7147 = invoke(stypy.reporting.localization.Localization(__file__, 463, 66), getitem___7146, int_7144)
                    
                    # Obtaining the member 'python_entity' of a type (line 463)
                    python_entity_7148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 66), subscript_call_result_7147, 'python_entity')
                    # Processing the call keyword arguments (line 463)
                    kwargs_7149 = {}
                    # Getting the type of 'str' (line 463)
                    str_7143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 62), 'str', False)
                    # Calling str(args, kwargs) (line 463)
                    str_call_result_7150 = invoke(stypy.reporting.localization.Localization(__file__, 463, 62), str_7143, *[python_entity_7148], **kwargs_7149)
                    
                    # Processing the call keyword arguments (line 460)
                    kwargs_7151 = {}
                    str_7133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 37), 'str', "Call to {0}{1} is invalid. Argument 1 requires a '{3}' but received a '{4}' \n\t{2}")
                    # Obtaining the member 'format' of a type (line 460)
                    format_7134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 37), str_7133, 'format')
                    # Calling format(args, kwargs) (line 460)
                    format_call_result_7152 = invoke(stypy.reporting.localization.Localization(__file__, 460, 37), format_7134, *[name_7135, arg_description_7136, usage_hint_7137, str_call_result_7142, str_call_result_7150], **kwargs_7151)
                    
                    # Processing the call keyword arguments (line 459)
                    # Getting the type of 'prints_msg' (line 464)
                    prints_msg_7153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 48), 'prints_msg', False)
                    keyword_7154 = prints_msg_7153
                    kwargs_7155 = {'prints_msg': keyword_7154}
                    # Getting the type of 'TypeError' (line 459)
                    TypeError_7131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 27), 'TypeError', False)
                    # Calling TypeError(args, kwargs) (line 459)
                    TypeError_call_result_7156 = invoke(stypy.reporting.localization.Localization(__file__, 459, 27), TypeError_7131, *[localization_7132, format_call_result_7152], **kwargs_7155)
                    
                    # Assigning a type to the variable 'stypy_return_type' (line 459)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 459, 20), 'stypy_return_type', TypeError_call_result_7156)
                    # SSA branch for the else part of an if statement (line 453)
                    module_type_store.open_ssa_branch('else')
                    
                    # Assigning a Call to a Name (line 466):
                    
                    # Assigning a Call to a Name (line 466):
                    
                    # Call to tuple(...): (line 466)
                    # Processing the call arguments (line 466)
                    
                    # Call to list(...): (line 466)
                    # Processing the call arguments (line 466)
                    
                    # Obtaining the type of the subscript
                    int_7159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 466, 58), 'int')
                    slice_7160 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 466, 48), int_7159, None, None)
                    # Getting the type of 'arg_types' (line 466)
                    arg_types_7161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 48), 'arg_types', False)
                    # Obtaining the member '__getitem__' of a type (line 466)
                    getitem___7162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 48), arg_types_7161, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 466)
                    subscript_call_result_7163 = invoke(stypy.reporting.localization.Localization(__file__, 466, 48), getitem___7162, slice_7160)
                    
                    # Processing the call keyword arguments (line 466)
                    kwargs_7164 = {}
                    # Getting the type of 'list' (line 466)
                    list_7158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 43), 'list', False)
                    # Calling list(args, kwargs) (line 466)
                    list_call_result_7165 = invoke(stypy.reporting.localization.Localization(__file__, 466, 43), list_7158, *[subscript_call_result_7163], **kwargs_7164)
                    
                    
                    # Call to values(...): (line 466)
                    # Processing the call keyword arguments (line 466)
                    kwargs_7168 = {}
                    # Getting the type of 'kwargs_types' (line 466)
                    kwargs_types_7166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 65), 'kwargs_types', False)
                    # Obtaining the member 'values' of a type (line 466)
                    values_7167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 65), kwargs_types_7166, 'values')
                    # Calling values(args, kwargs) (line 466)
                    values_call_result_7169 = invoke(stypy.reporting.localization.Localization(__file__, 466, 65), values_7167, *[], **kwargs_7168)
                    
                    # Applying the binary operator '+' (line 466)
                    result_add_7170 = python_operator(stypy.reporting.localization.Localization(__file__, 466, 43), '+', list_call_result_7165, values_call_result_7169)
                    
                    # Processing the call keyword arguments (line 466)
                    kwargs_7171 = {}
                    # Getting the type of 'tuple' (line 466)
                    tuple_7157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 37), 'tuple', False)
                    # Calling tuple(args, kwargs) (line 466)
                    tuple_call_result_7172 = invoke(stypy.reporting.localization.Localization(__file__, 466, 37), tuple_7157, *[result_add_7170], **kwargs_7171)
                    
                    # Assigning a type to the variable 'argument_types' (line 466)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 466, 20), 'argument_types', tuple_call_result_7172)
                    # SSA join for if statement (line 453)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for if statement (line 451)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 449)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Type idiom detected: calculating its left and rigth part (line 469)
        # Getting the type of 'argument_types' (line 469)
        argument_types_7173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 11), 'argument_types')
        # Getting the type of 'None' (line 469)
        None_7174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 29), 'None')
        
        (may_be_7175, more_types_in_union_7176) = may_be_none(argument_types_7173, None_7174)

        if may_be_7175:

            if more_types_in_union_7176:
                # Runtime conditional SSA (line 469)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 470):
            
            # Assigning a Call to a Name (line 470):
            
            # Call to tuple(...): (line 470)
            # Processing the call arguments (line 470)
            
            # Call to list(...): (line 470)
            # Processing the call arguments (line 470)
            # Getting the type of 'arg_types' (line 470)
            arg_types_7179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 40), 'arg_types', False)
            # Processing the call keyword arguments (line 470)
            kwargs_7180 = {}
            # Getting the type of 'list' (line 470)
            list_7178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 35), 'list', False)
            # Calling list(args, kwargs) (line 470)
            list_call_result_7181 = invoke(stypy.reporting.localization.Localization(__file__, 470, 35), list_7178, *[arg_types_7179], **kwargs_7180)
            
            # Processing the call keyword arguments (line 470)
            kwargs_7182 = {}
            # Getting the type of 'tuple' (line 470)
            tuple_7177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 29), 'tuple', False)
            # Calling tuple(args, kwargs) (line 470)
            tuple_call_result_7183 = invoke(stypy.reporting.localization.Localization(__file__, 470, 29), tuple_7177, *[list_call_result_7181], **kwargs_7182)
            
            # Assigning a type to the variable 'argument_types' (line 470)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 470, 12), 'argument_types', tuple_call_result_7183)

            if more_types_in_union_7176:
                # SSA join for if statement (line 469)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 472):
        
        # Assigning a Call to a Name (line 472):
        
        # Call to len(...): (line 472)
        # Processing the call arguments (line 472)
        # Getting the type of 'argument_types' (line 472)
        argument_types_7185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 25), 'argument_types', False)
        # Processing the call keyword arguments (line 472)
        kwargs_7186 = {}
        # Getting the type of 'len' (line 472)
        len_7184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 21), 'len', False)
        # Calling len(args, kwargs) (line 472)
        len_call_result_7187 = invoke(stypy.reporting.localization.Localization(__file__, 472, 21), len_7184, *[argument_types_7185], **kwargs_7186)
        
        # Assigning a type to the variable 'call_arity' (line 472)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 472, 8), 'call_arity', len_call_result_7187)
        
        # Getting the type of 'rules' (line 475)
        rules_7188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 46), 'rules')
        # Assigning a type to the variable 'rules_7188' (line 475)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 475, 8), 'rules_7188', rules_7188)
        # Testing if the for loop is going to be iterated (line 475)
        # Testing the type of a for loop iterable (line 475)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 475, 8), rules_7188)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 475, 8), rules_7188):
            # Getting the type of the for loop variable (line 475)
            for_loop_var_7189 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 475, 8), rules_7188)
            # Assigning a type to the variable 'params_in_rules' (line 475)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 475, 8), 'params_in_rules', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 475, 8), for_loop_var_7189, 2, 0))
            # Assigning a type to the variable 'return_type' (line 475)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 475, 8), 'return_type', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 475, 8), for_loop_var_7189, 2, 1))
            # SSA begins for a for statement (line 475)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Evaluating a boolean operation
            
            
            # Call to len(...): (line 477)
            # Processing the call arguments (line 477)
            # Getting the type of 'params_in_rules' (line 477)
            params_in_rules_7191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 19), 'params_in_rules', False)
            # Processing the call keyword arguments (line 477)
            kwargs_7192 = {}
            # Getting the type of 'len' (line 477)
            len_7190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 15), 'len', False)
            # Calling len(args, kwargs) (line 477)
            len_call_result_7193 = invoke(stypy.reporting.localization.Localization(__file__, 477, 15), len_7190, *[params_in_rules_7191], **kwargs_7192)
            
            # Getting the type of 'call_arity' (line 477)
            call_arity_7194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 39), 'call_arity')
            # Applying the binary operator '==' (line 477)
            result_eq_7195 = python_operator(stypy.reporting.localization.Localization(__file__, 477, 15), '==', len_call_result_7193, call_arity_7194)
            
            
            # Call to __has_varargs_in_rule_params(...): (line 477)
            # Processing the call arguments (line 477)
            # Getting the type of 'params_in_rules' (line 477)
            params_in_rules_7198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 87), 'params_in_rules', False)
            # Processing the call keyword arguments (line 477)
            kwargs_7199 = {}
            # Getting the type of 'self' (line 477)
            self_7196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 53), 'self', False)
            # Obtaining the member '__has_varargs_in_rule_params' of a type (line 477)
            has_varargs_in_rule_params_7197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 477, 53), self_7196, '__has_varargs_in_rule_params')
            # Calling __has_varargs_in_rule_params(args, kwargs) (line 477)
            has_varargs_in_rule_params_call_result_7200 = invoke(stypy.reporting.localization.Localization(__file__, 477, 53), has_varargs_in_rule_params_7197, *[params_in_rules_7198], **kwargs_7199)
            
            # Applying the binary operator 'or' (line 477)
            result_or_keyword_7201 = python_operator(stypy.reporting.localization.Localization(__file__, 477, 15), 'or', result_eq_7195, has_varargs_in_rule_params_call_result_7200)
            
            # Testing if the type of an if condition is none (line 477)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 477, 12), result_or_keyword_7201):
                pass
            else:
                
                # Testing the type of an if condition (line 477)
                if_condition_7202 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 477, 12), result_or_keyword_7201)
                # Assigning a type to the variable 'if_condition_7202' (line 477)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 477, 12), 'if_condition_7202', if_condition_7202)
                # SSA begins for if statement (line 477)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to __compare(...): (line 479)
                # Processing the call arguments (line 479)
                # Getting the type of 'params_in_rules' (line 479)
                params_in_rules_7205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 34), 'params_in_rules', False)
                # Getting the type of 'argument_types' (line 479)
                argument_types_7206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 51), 'argument_types', False)
                # Processing the call keyword arguments (line 479)
                kwargs_7207 = {}
                # Getting the type of 'self' (line 479)
                self_7203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 19), 'self', False)
                # Obtaining the member '__compare' of a type (line 479)
                compare_7204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 479, 19), self_7203, '__compare')
                # Calling __compare(args, kwargs) (line 479)
                compare_call_result_7208 = invoke(stypy.reporting.localization.Localization(__file__, 479, 19), compare_7204, *[params_in_rules_7205, argument_types_7206], **kwargs_7207)
                
                # Testing if the type of an if condition is none (line 479)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 479, 16), compare_call_result_7208):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 479)
                    if_condition_7209 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 479, 16), compare_call_result_7208)
                    # Assigning a type to the variable 'if_condition_7209' (line 479)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 479, 16), 'if_condition_7209', if_condition_7209)
                    # SSA begins for if statement (line 479)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Call to __dependent_type_in_rule_params(...): (line 481)
                    # Processing the call arguments (line 481)
                    # Getting the type of 'params_in_rules' (line 481)
                    params_in_rules_7212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 60), 'params_in_rules', False)
                    # Processing the call keyword arguments (line 481)
                    kwargs_7213 = {}
                    # Getting the type of 'self' (line 481)
                    self_7210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 23), 'self', False)
                    # Obtaining the member '__dependent_type_in_rule_params' of a type (line 481)
                    dependent_type_in_rule_params_7211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 481, 23), self_7210, '__dependent_type_in_rule_params')
                    # Calling __dependent_type_in_rule_params(args, kwargs) (line 481)
                    dependent_type_in_rule_params_call_result_7214 = invoke(stypy.reporting.localization.Localization(__file__, 481, 23), dependent_type_in_rule_params_7211, *[params_in_rules_7212], **kwargs_7213)
                    
                    # Testing if the type of an if condition is none (line 481)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 481, 20), dependent_type_in_rule_params_call_result_7214):
                        
                        # Call to __create_return_type(...): (line 509)
                        # Processing the call arguments (line 509)
                        # Getting the type of 'localization' (line 509)
                        localization_7273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 57), 'localization', False)
                        # Getting the type of 'return_type' (line 509)
                        return_type_7274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 71), 'return_type', False)
                        # Getting the type of 'argument_types' (line 509)
                        argument_types_7275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 84), 'argument_types', False)
                        # Processing the call keyword arguments (line 509)
                        kwargs_7276 = {}
                        # Getting the type of 'self' (line 509)
                        self_7271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 31), 'self', False)
                        # Obtaining the member '__create_return_type' of a type (line 509)
                        create_return_type_7272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 509, 31), self_7271, '__create_return_type')
                        # Calling __create_return_type(args, kwargs) (line 509)
                        create_return_type_call_result_7277 = invoke(stypy.reporting.localization.Localization(__file__, 509, 31), create_return_type_7272, *[localization_7273, return_type_7274, argument_types_7275], **kwargs_7276)
                        
                        # Assigning a type to the variable 'stypy_return_type' (line 509)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 509, 24), 'stypy_return_type', create_return_type_call_result_7277)
                    else:
                        
                        # Testing the type of an if condition (line 481)
                        if_condition_7215 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 481, 20), dependent_type_in_rule_params_call_result_7214)
                        # Assigning a type to the variable 'if_condition_7215' (line 481)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 481, 20), 'if_condition_7215', if_condition_7215)
                        # SSA begins for if statement (line 481)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Assigning a Call to a Tuple (line 483):
                        
                        # Assigning a Call to a Name:
                        
                        # Call to invoke_dependent_rules(...): (line 483)
                        # Processing the call arguments (line 483)
                        # Getting the type of 'localization' (line 484)
                        localization_7218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 28), 'localization', False)
                        # Getting the type of 'params_in_rules' (line 484)
                        params_in_rules_7219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 42), 'params_in_rules', False)
                        # Getting the type of 'argument_types' (line 484)
                        argument_types_7220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 59), 'argument_types', False)
                        # Processing the call keyword arguments (line 483)
                        kwargs_7221 = {}
                        # Getting the type of 'self' (line 483)
                        self_7216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 86), 'self', False)
                        # Obtaining the member 'invoke_dependent_rules' of a type (line 483)
                        invoke_dependent_rules_7217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 483, 86), self_7216, 'invoke_dependent_rules')
                        # Calling invoke_dependent_rules(args, kwargs) (line 483)
                        invoke_dependent_rules_call_result_7222 = invoke(stypy.reporting.localization.Localization(__file__, 483, 86), invoke_dependent_rules_7217, *[localization_7218, params_in_rules_7219, argument_types_7220], **kwargs_7221)
                        
                        # Assigning a type to the variable 'call_assignment_6193' (line 483)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 483, 24), 'call_assignment_6193', invoke_dependent_rules_call_result_7222)
                        
                        # Assigning a Call to a Name (line 483):
                        
                        # Call to stypy_get_value_from_tuple(...):
                        # Processing the call arguments
                        # Getting the type of 'call_assignment_6193' (line 483)
                        call_assignment_6193_7223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 24), 'call_assignment_6193', False)
                        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                        stypy_get_value_from_tuple_call_result_7224 = stypy_get_value_from_tuple(call_assignment_6193_7223, 4, 0)
                        
                        # Assigning a type to the variable 'call_assignment_6194' (line 483)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 483, 24), 'call_assignment_6194', stypy_get_value_from_tuple_call_result_7224)
                        
                        # Assigning a Name to a Name (line 483):
                        # Getting the type of 'call_assignment_6194' (line 483)
                        call_assignment_6194_7225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 24), 'call_assignment_6194')
                        # Assigning a type to the variable 'correct' (line 483)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 483, 24), 'correct', call_assignment_6194_7225)
                        
                        # Assigning a Call to a Name (line 483):
                        
                        # Call to stypy_get_value_from_tuple(...):
                        # Processing the call arguments
                        # Getting the type of 'call_assignment_6193' (line 483)
                        call_assignment_6193_7226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 24), 'call_assignment_6193', False)
                        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                        stypy_get_value_from_tuple_call_result_7227 = stypy_get_value_from_tuple(call_assignment_6193_7226, 4, 1)
                        
                        # Assigning a type to the variable 'call_assignment_6195' (line 483)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 483, 24), 'call_assignment_6195', stypy_get_value_from_tuple_call_result_7227)
                        
                        # Assigning a Name to a Name (line 483):
                        # Getting the type of 'call_assignment_6195' (line 483)
                        call_assignment_6195_7228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 24), 'call_assignment_6195')
                        # Assigning a type to the variable 'equivalent_rule' (line 483)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 483, 33), 'equivalent_rule', call_assignment_6195_7228)
                        
                        # Assigning a Call to a Name (line 483):
                        
                        # Call to stypy_get_value_from_tuple(...):
                        # Processing the call arguments
                        # Getting the type of 'call_assignment_6193' (line 483)
                        call_assignment_6193_7229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 24), 'call_assignment_6193', False)
                        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                        stypy_get_value_from_tuple_call_result_7230 = stypy_get_value_from_tuple(call_assignment_6193_7229, 4, 2)
                        
                        # Assigning a type to the variable 'call_assignment_6196' (line 483)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 483, 24), 'call_assignment_6196', stypy_get_value_from_tuple_call_result_7230)
                        
                        # Assigning a Name to a Name (line 483):
                        # Getting the type of 'call_assignment_6196' (line 483)
                        call_assignment_6196_7231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 24), 'call_assignment_6196')
                        # Assigning a type to the variable 'needs_reevaluation' (line 483)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 483, 50), 'needs_reevaluation', call_assignment_6196_7231)
                        
                        # Assigning a Call to a Name (line 483):
                        
                        # Call to stypy_get_value_from_tuple(...):
                        # Processing the call arguments
                        # Getting the type of 'call_assignment_6193' (line 483)
                        call_assignment_6193_7232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 24), 'call_assignment_6193', False)
                        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                        stypy_get_value_from_tuple_call_result_7233 = stypy_get_value_from_tuple(call_assignment_6193_7232, 4, 3)
                        
                        # Assigning a type to the variable 'call_assignment_6197' (line 483)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 483, 24), 'call_assignment_6197', stypy_get_value_from_tuple_call_result_7233)
                        
                        # Assigning a Name to a Name (line 483):
                        # Getting the type of 'call_assignment_6197' (line 483)
                        call_assignment_6197_7234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 24), 'call_assignment_6197')
                        # Assigning a type to the variable 'invokation_rt' (line 483)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 483, 70), 'invokation_rt', call_assignment_6197_7234)
                        # Getting the type of 'correct' (line 486)
                        correct_7235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 27), 'correct')
                        # Testing if the type of an if condition is none (line 486)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 486, 24), correct_7235):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 486)
                            if_condition_7236 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 486, 24), correct_7235)
                            # Assigning a type to the variable 'if_condition_7236' (line 486)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 486, 24), 'if_condition_7236', if_condition_7236)
                            # SSA begins for if statement (line 486)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Getting the type of 'needs_reevaluation' (line 488)
                            needs_reevaluation_7237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 35), 'needs_reevaluation')
                            # Applying the 'not' unary operator (line 488)
                            result_not__7238 = python_operator(stypy.reporting.localization.Localization(__file__, 488, 31), 'not', needs_reevaluation_7237)
                            
                            # Testing if the type of an if condition is none (line 488)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 488, 28), result_not__7238):
                                
                                # Getting the type of 'rules' (line 502)
                                rules_7258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 72), 'rules')
                                # Assigning a type to the variable 'rules_7258' (line 502)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 502, 32), 'rules_7258', rules_7258)
                                # Testing if the for loop is going to be iterated (line 502)
                                # Testing the type of a for loop iterable (line 502)
                                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 502, 32), rules_7258)

                                if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 502, 32), rules_7258):
                                    # Getting the type of the for loop variable (line 502)
                                    for_loop_var_7259 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 502, 32), rules_7258)
                                    # Assigning a type to the variable 'params_in_rules2' (line 502)
                                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 502, 32), 'params_in_rules2', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 502, 32), for_loop_var_7259, 2, 0))
                                    # Assigning a type to the variable 'return_type2' (line 502)
                                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 502, 32), 'return_type2', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 502, 32), for_loop_var_7259, 2, 1))
                                    # SSA begins for a for statement (line 502)
                                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                                    
                                    # Getting the type of 'params_in_rules2' (line 504)
                                    params_in_rules2_7260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 39), 'params_in_rules2')
                                    # Getting the type of 'equivalent_rule' (line 504)
                                    equivalent_rule_7261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 59), 'equivalent_rule')
                                    # Applying the binary operator '==' (line 504)
                                    result_eq_7262 = python_operator(stypy.reporting.localization.Localization(__file__, 504, 39), '==', params_in_rules2_7260, equivalent_rule_7261)
                                    
                                    # Testing if the type of an if condition is none (line 504)

                                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 504, 36), result_eq_7262):
                                        pass
                                    else:
                                        
                                        # Testing the type of an if condition (line 504)
                                        if_condition_7263 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 504, 36), result_eq_7262)
                                        # Assigning a type to the variable 'if_condition_7263' (line 504)
                                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 504, 36), 'if_condition_7263', if_condition_7263)
                                        # SSA begins for if statement (line 504)
                                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                        
                                        # Call to __create_return_type(...): (line 506)
                                        # Processing the call arguments (line 506)
                                        # Getting the type of 'localization' (line 506)
                                        localization_7266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 73), 'localization', False)
                                        # Getting the type of 'return_type2' (line 506)
                                        return_type2_7267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 87), 'return_type2', False)
                                        # Getting the type of 'argument_types' (line 506)
                                        argument_types_7268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 101), 'argument_types', False)
                                        # Processing the call keyword arguments (line 506)
                                        kwargs_7269 = {}
                                        # Getting the type of 'self' (line 506)
                                        self_7264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 47), 'self', False)
                                        # Obtaining the member '__create_return_type' of a type (line 506)
                                        create_return_type_7265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 506, 47), self_7264, '__create_return_type')
                                        # Calling __create_return_type(args, kwargs) (line 506)
                                        create_return_type_call_result_7270 = invoke(stypy.reporting.localization.Localization(__file__, 506, 47), create_return_type_7265, *[localization_7266, return_type2_7267, argument_types_7268], **kwargs_7269)
                                        
                                        # Assigning a type to the variable 'stypy_return_type' (line 506)
                                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 506, 40), 'stypy_return_type', create_return_type_call_result_7270)
                                        # SSA join for if statement (line 504)
                                        module_type_store = module_type_store.join_ssa_context()
                                        

                                    # SSA join for a for statement
                                    module_type_store = module_type_store.join_ssa_context()

                                
                            else:
                                
                                # Testing the type of an if condition (line 488)
                                if_condition_7239 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 488, 28), result_not__7238)
                                # Assigning a type to the variable 'if_condition_7239' (line 488)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 488, 28), 'if_condition_7239', if_condition_7239)
                                # SSA begins for if statement (line 488)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Type idiom detected: calculating its left and rigth part (line 491)
                                # Getting the type of 'invokation_rt' (line 491)
                                invokation_rt_7240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 32), 'invokation_rt')
                                # Getting the type of 'None' (line 491)
                                None_7241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 56), 'None')
                                
                                (may_be_7242, more_types_in_union_7243) = may_not_be_none(invokation_rt_7240, None_7241)

                                if may_be_7242:

                                    if more_types_in_union_7243:
                                        # Runtime conditional SSA (line 491)
                                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                                    else:
                                        module_type_store = module_type_store

                                    
                                    # Call to __create_return_type(...): (line 492)
                                    # Processing the call arguments (line 492)
                                    # Getting the type of 'localization' (line 492)
                                    localization_7246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 69), 'localization', False)
                                    # Getting the type of 'invokation_rt' (line 492)
                                    invokation_rt_7247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 83), 'invokation_rt', False)
                                    # Getting the type of 'argument_types' (line 492)
                                    argument_types_7248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 98), 'argument_types', False)
                                    # Processing the call keyword arguments (line 492)
                                    kwargs_7249 = {}
                                    # Getting the type of 'self' (line 492)
                                    self_7244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 43), 'self', False)
                                    # Obtaining the member '__create_return_type' of a type (line 492)
                                    create_return_type_7245 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 492, 43), self_7244, '__create_return_type')
                                    # Calling __create_return_type(args, kwargs) (line 492)
                                    create_return_type_call_result_7250 = invoke(stypy.reporting.localization.Localization(__file__, 492, 43), create_return_type_7245, *[localization_7246, invokation_rt_7247, argument_types_7248], **kwargs_7249)
                                    
                                    # Assigning a type to the variable 'stypy_return_type' (line 492)
                                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 492, 36), 'stypy_return_type', create_return_type_call_result_7250)

                                    if more_types_in_union_7243:
                                        # SSA join for if statement (line 491)
                                        module_type_store = module_type_store.join_ssa_context()


                                
                                
                                # Call to __create_return_type(...): (line 497)
                                # Processing the call arguments (line 497)
                                # Getting the type of 'localization' (line 497)
                                localization_7253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 65), 'localization', False)
                                # Getting the type of 'return_type' (line 497)
                                return_type_7254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 79), 'return_type', False)
                                # Getting the type of 'argument_types' (line 497)
                                argument_types_7255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 92), 'argument_types', False)
                                # Processing the call keyword arguments (line 497)
                                kwargs_7256 = {}
                                # Getting the type of 'self' (line 497)
                                self_7251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 39), 'self', False)
                                # Obtaining the member '__create_return_type' of a type (line 497)
                                create_return_type_7252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 497, 39), self_7251, '__create_return_type')
                                # Calling __create_return_type(args, kwargs) (line 497)
                                create_return_type_call_result_7257 = invoke(stypy.reporting.localization.Localization(__file__, 497, 39), create_return_type_7252, *[localization_7253, return_type_7254, argument_types_7255], **kwargs_7256)
                                
                                # Assigning a type to the variable 'stypy_return_type' (line 497)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 497, 32), 'stypy_return_type', create_return_type_call_result_7257)
                                # SSA branch for the else part of an if statement (line 488)
                                module_type_store.open_ssa_branch('else')
                                
                                # Getting the type of 'rules' (line 502)
                                rules_7258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 72), 'rules')
                                # Assigning a type to the variable 'rules_7258' (line 502)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 502, 32), 'rules_7258', rules_7258)
                                # Testing if the for loop is going to be iterated (line 502)
                                # Testing the type of a for loop iterable (line 502)
                                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 502, 32), rules_7258)

                                if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 502, 32), rules_7258):
                                    # Getting the type of the for loop variable (line 502)
                                    for_loop_var_7259 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 502, 32), rules_7258)
                                    # Assigning a type to the variable 'params_in_rules2' (line 502)
                                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 502, 32), 'params_in_rules2', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 502, 32), for_loop_var_7259, 2, 0))
                                    # Assigning a type to the variable 'return_type2' (line 502)
                                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 502, 32), 'return_type2', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 502, 32), for_loop_var_7259, 2, 1))
                                    # SSA begins for a for statement (line 502)
                                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                                    
                                    # Getting the type of 'params_in_rules2' (line 504)
                                    params_in_rules2_7260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 39), 'params_in_rules2')
                                    # Getting the type of 'equivalent_rule' (line 504)
                                    equivalent_rule_7261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 59), 'equivalent_rule')
                                    # Applying the binary operator '==' (line 504)
                                    result_eq_7262 = python_operator(stypy.reporting.localization.Localization(__file__, 504, 39), '==', params_in_rules2_7260, equivalent_rule_7261)
                                    
                                    # Testing if the type of an if condition is none (line 504)

                                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 504, 36), result_eq_7262):
                                        pass
                                    else:
                                        
                                        # Testing the type of an if condition (line 504)
                                        if_condition_7263 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 504, 36), result_eq_7262)
                                        # Assigning a type to the variable 'if_condition_7263' (line 504)
                                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 504, 36), 'if_condition_7263', if_condition_7263)
                                        # SSA begins for if statement (line 504)
                                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                        
                                        # Call to __create_return_type(...): (line 506)
                                        # Processing the call arguments (line 506)
                                        # Getting the type of 'localization' (line 506)
                                        localization_7266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 73), 'localization', False)
                                        # Getting the type of 'return_type2' (line 506)
                                        return_type2_7267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 87), 'return_type2', False)
                                        # Getting the type of 'argument_types' (line 506)
                                        argument_types_7268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 101), 'argument_types', False)
                                        # Processing the call keyword arguments (line 506)
                                        kwargs_7269 = {}
                                        # Getting the type of 'self' (line 506)
                                        self_7264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 47), 'self', False)
                                        # Obtaining the member '__create_return_type' of a type (line 506)
                                        create_return_type_7265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 506, 47), self_7264, '__create_return_type')
                                        # Calling __create_return_type(args, kwargs) (line 506)
                                        create_return_type_call_result_7270 = invoke(stypy.reporting.localization.Localization(__file__, 506, 47), create_return_type_7265, *[localization_7266, return_type2_7267, argument_types_7268], **kwargs_7269)
                                        
                                        # Assigning a type to the variable 'stypy_return_type' (line 506)
                                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 506, 40), 'stypy_return_type', create_return_type_call_result_7270)
                                        # SSA join for if statement (line 504)
                                        module_type_store = module_type_store.join_ssa_context()
                                        

                                    # SSA join for a for statement
                                    module_type_store = module_type_store.join_ssa_context()

                                
                                # SSA join for if statement (line 488)
                                module_type_store = module_type_store.join_ssa_context()
                                

                            # SSA join for if statement (line 486)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        # SSA branch for the else part of an if statement (line 481)
                        module_type_store.open_ssa_branch('else')
                        
                        # Call to __create_return_type(...): (line 509)
                        # Processing the call arguments (line 509)
                        # Getting the type of 'localization' (line 509)
                        localization_7273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 57), 'localization', False)
                        # Getting the type of 'return_type' (line 509)
                        return_type_7274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 71), 'return_type', False)
                        # Getting the type of 'argument_types' (line 509)
                        argument_types_7275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 84), 'argument_types', False)
                        # Processing the call keyword arguments (line 509)
                        kwargs_7276 = {}
                        # Getting the type of 'self' (line 509)
                        self_7271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 31), 'self', False)
                        # Obtaining the member '__create_return_type' of a type (line 509)
                        create_return_type_7272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 509, 31), self_7271, '__create_return_type')
                        # Calling __create_return_type(args, kwargs) (line 509)
                        create_return_type_call_result_7277 = invoke(stypy.reporting.localization.Localization(__file__, 509, 31), create_return_type_7272, *[localization_7273, return_type_7274, argument_types_7275], **kwargs_7276)
                        
                        # Assigning a type to the variable 'stypy_return_type' (line 509)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 509, 24), 'stypy_return_type', create_return_type_call_result_7277)
                        # SSA join for if statement (line 481)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for if statement (line 479)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for if statement (line 477)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Assigning a Call to a Name (line 512):
        
        # Assigning a Call to a Name (line 512):
        
        # Call to __format_admitted_params(...): (line 512)
        # Processing the call arguments (line 512)
        # Getting the type of 'name' (line 512)
        name_7280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 51), 'name', False)
        # Getting the type of 'rules' (line 512)
        rules_7281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 57), 'rules', False)
        # Getting the type of 'argument_types' (line 512)
        argument_types_7282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 64), 'argument_types', False)
        
        # Call to len(...): (line 512)
        # Processing the call arguments (line 512)
        # Getting the type of 'argument_types' (line 512)
        argument_types_7284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 84), 'argument_types', False)
        # Processing the call keyword arguments (line 512)
        kwargs_7285 = {}
        # Getting the type of 'len' (line 512)
        len_7283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 80), 'len', False)
        # Calling len(args, kwargs) (line 512)
        len_call_result_7286 = invoke(stypy.reporting.localization.Localization(__file__, 512, 80), len_7283, *[argument_types_7284], **kwargs_7285)
        
        # Processing the call keyword arguments (line 512)
        kwargs_7287 = {}
        # Getting the type of 'self' (line 512)
        self_7278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 21), 'self', False)
        # Obtaining the member '__format_admitted_params' of a type (line 512)
        format_admitted_params_7279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 512, 21), self_7278, '__format_admitted_params')
        # Calling __format_admitted_params(args, kwargs) (line 512)
        format_admitted_params_call_result_7288 = invoke(stypy.reporting.localization.Localization(__file__, 512, 21), format_admitted_params_7279, *[name_7280, rules_7281, argument_types_7282, len_call_result_7286], **kwargs_7287)
        
        # Assigning a type to the variable 'usage_hint' (line 512)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 512, 8), 'usage_hint', format_admitted_params_call_result_7288)
        
        # Assigning a Call to a Name (line 514):
        
        # Assigning a Call to a Name (line 514):
        
        # Call to str(...): (line 514)
        # Processing the call arguments (line 514)
        # Getting the type of 'argument_types' (line 514)
        argument_types_7290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 30), 'argument_types', False)
        # Processing the call keyword arguments (line 514)
        kwargs_7291 = {}
        # Getting the type of 'str' (line 514)
        str_7289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 26), 'str', False)
        # Calling str(args, kwargs) (line 514)
        str_call_result_7292 = invoke(stypy.reporting.localization.Localization(__file__, 514, 26), str_7289, *[argument_types_7290], **kwargs_7291)
        
        # Assigning a type to the variable 'arg_description' (line 514)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 514, 8), 'arg_description', str_call_result_7292)
        
        # Assigning a Call to a Name (line 515):
        
        # Assigning a Call to a Name (line 515):
        
        # Call to replace(...): (line 515)
        # Processing the call arguments (line 515)
        str_7295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 50), 'str', ',)')
        str_7296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 56), 'str', ')')
        # Processing the call keyword arguments (line 515)
        kwargs_7297 = {}
        # Getting the type of 'arg_description' (line 515)
        arg_description_7293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 26), 'arg_description', False)
        # Obtaining the member 'replace' of a type (line 515)
        replace_7294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 515, 26), arg_description_7293, 'replace')
        # Calling replace(args, kwargs) (line 515)
        replace_call_result_7298 = invoke(stypy.reporting.localization.Localization(__file__, 515, 26), replace_7294, *[str_7295, str_7296], **kwargs_7297)
        
        # Assigning a type to the variable 'arg_description' (line 515)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 515, 8), 'arg_description', replace_call_result_7298)
        
        # Call to TypeError(...): (line 516)
        # Processing the call arguments (line 516)
        # Getting the type of 'localization' (line 516)
        localization_7300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 25), 'localization', False)
        
        # Call to format(...): (line 516)
        # Processing the call arguments (line 516)
        # Getting the type of 'name' (line 516)
        name_7303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 82), 'name', False)
        # Getting the type of 'arg_description' (line 516)
        arg_description_7304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 88), 'arg_description', False)
        # Getting the type of 'usage_hint' (line 517)
        usage_hint_7305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 82), 'usage_hint', False)
        # Processing the call keyword arguments (line 516)
        kwargs_7306 = {}
        str_7301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 39), 'str', 'Call to {0}{1} is invalid.\n\t{2}')
        # Obtaining the member 'format' of a type (line 516)
        format_7302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 516, 39), str_7301, 'format')
        # Calling format(args, kwargs) (line 516)
        format_call_result_7307 = invoke(stypy.reporting.localization.Localization(__file__, 516, 39), format_7302, *[name_7303, arg_description_7304, usage_hint_7305], **kwargs_7306)
        
        # Processing the call keyword arguments (line 516)
        # Getting the type of 'prints_msg' (line 517)
        prints_msg_7308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 106), 'prints_msg', False)
        keyword_7309 = prints_msg_7308
        kwargs_7310 = {'prints_msg': keyword_7309}
        # Getting the type of 'TypeError' (line 516)
        TypeError_7299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 15), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 516)
        TypeError_call_result_7311 = invoke(stypy.reporting.localization.Localization(__file__, 516, 15), TypeError_7299, *[localization_7300, format_call_result_7307], **kwargs_7310)
        
        # Assigning a type to the variable 'stypy_return_type' (line 516)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 516, 8), 'stypy_return_type', TypeError_call_result_7311)
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 422)
        stypy_return_type_7312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_7312)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_7312


# Assigning a type to the variable 'TypeRuleCallHandler' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'TypeRuleCallHandler', TypeRuleCallHandler)

# Assigning a Call to a Name (line 20):

# Call to dict(...): (line 20)
# Processing the call keyword arguments (line 20)
kwargs_7314 = {}
# Getting the type of 'dict' (line 20)
dict_7313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 22), 'dict', False)
# Calling dict(args, kwargs) (line 20)
dict_call_result_7315 = invoke(stypy.reporting.localization.Localization(__file__, 20, 22), dict_7313, *[], **kwargs_7314)

# Getting the type of 'TypeRuleCallHandler'
TypeRuleCallHandler_7316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TypeRuleCallHandler')
# Setting the type of the member 'type_rule_cache' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TypeRuleCallHandler_7316, 'type_rule_cache', dict_call_result_7315)

# Assigning a Call to a Name (line 23):

# Call to dict(...): (line 23)
# Processing the call keyword arguments (line 23)
kwargs_7318 = {}
# Getting the type of 'dict' (line 23)
dict_7317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 34), 'dict', False)
# Calling dict(args, kwargs) (line 23)
dict_call_result_7319 = invoke(stypy.reporting.localization.Localization(__file__, 23, 34), dict_7317, *[], **kwargs_7318)

# Getting the type of 'TypeRuleCallHandler'
TypeRuleCallHandler_7320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TypeRuleCallHandler')
# Setting the type of the member 'unavailable_type_rule_cache' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TypeRuleCallHandler_7320, 'unavailable_type_rule_cache', dict_call_result_7319)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
