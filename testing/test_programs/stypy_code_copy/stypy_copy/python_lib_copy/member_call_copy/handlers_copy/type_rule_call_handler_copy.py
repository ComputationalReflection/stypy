import os
import sys
import inspect

from call_handler_copy import CallHandler
from ....python_lib_copy.python_types_copy import type_inference_copy
from .... import stypy_parameters_copy
from ....python_lib_copy.type_rules_copy.type_groups_copy.type_groups_copy import *
from ....python_lib_copy.type_rules_copy.type_groups_copy.type_group_copy import BaseTypeGroup


class TypeRuleCallHandler(CallHandler):
    """
    This call handler uses type rule files (Python files with a special structure) to determine acceptable parameters
    and return types for the calls of a certain module/class and its callable members. The handler dynamically search,
    load and use these rule files to resolve calls.
    """

    # Cache of found rule files
    type_rule_cache = dict()

    # Cache of not found rule files (to improve performance)
    unavailable_type_rule_cache = dict()

    @staticmethod
    def __rule_files(parent_name, entity_name):
        """
        For a call to parent_name.entity_name(...), compose the name of the type rule file that will correspond to the
        entity or its parent, to look inside any of them for suitable rules to apply
        :param parent_name: Parent entity (module/class) name
        :param entity_name: Callable entity (function/method) name
        :return: A tuple of (name of the rule file of the parent, name of the type rule of the entity)
        """
        parent_type_rule_file = stypy_parameters_copy.ROOT_PATH + stypy_parameters_copy.RULE_FILE_PATH + parent_name + "/" \
                                + parent_name + stypy_parameters_copy.type_rule_file_postfix + ".py"

        own_type_rule_file = stypy_parameters_copy.ROOT_PATH + stypy_parameters_copy.RULE_FILE_PATH + parent_name + "/" \
                             + entity_name.split('.')[-1] + "/" + entity_name.split('.')[
                                 -1] + stypy_parameters_copy.type_rule_file_postfix + ".py"

        return parent_type_rule_file, own_type_rule_file

    @staticmethod
    def __dependent_type_in_rule_params(params):
        """
        Check if a list of params has dependent types: Types that have to be called somewhat in order to obtain the
        real type they represent.
        :param params: List of types
        :return: bool
        """
        return len(filter(lambda par: isinstance(par, DependentType), params)) > 0

    @staticmethod
    def __has_varargs_in_rule_params(params):
        """
        Check if a list of params has variable number of arguments
        :param params: List of types
        :return: bool
        """
        return len(filter(lambda par: isinstance(par, VarArgType), params)) > 0

    @staticmethod
    def __get_arguments(argument_tuple, current_pos, rule_arity):
        """
        Obtain a list composed by the arguments present in argument_tuple, except the one in current_pos limited
        to rule_arity size. This is used when invoking dependent rules
        :param argument_tuple:
        :param current_pos:
        :param rule_arity:
        :return:
        """
        if rule_arity == 0:
            return []

        temp_list = []
        for i in range(len(argument_tuple)):
            if not i == current_pos:
                temp_list.append(argument_tuple[i])

        return tuple(temp_list[0:rule_arity])

    def invoke_dependent_rules(self, localization, rule_params, arguments):
        """
        As we said, some rules may contain special types called DependentTypes. These types have to be invoked in
        order to check that the rule matches with the call or other necessary operations. Dependent types may have
        several forms, and are called with all the arguments that are checked against the type rule except the one
        that matches de dependent type, limited by the Dependent type declared rule arity. For example a Dependent
        Type may be defined like this (see type_groups.py for all the Dependent types defined):

        Overloads__eq__ = HasMember("__eq__", DynamicType, 1)

        This means that Overloads__eq__ matches with all the objects that has a method named __eq__ that has no
        predefined return type and an arity of 1 parameter. On the other hand, a type rule may be defined like this:

        ((Overloads__eq__, AnyType), DynamicType)

        This means that the type rule matches with a call that has a first argument which overloads the method
        __eq__ and any kind of second arguments. Although __eq__ is a method that should return bool (is the ==
        operator) this is not compulsory in Python, the __eq__ method may return anything and this anything will be
        the result of the rule. So we have to call __eq__ with the second argument (all the arguments but the one
        that matches with the DependentType limited to the declared dependent type arity), capture and return the
        result. This is basically the functionality of this method.

        Note that invocation to a method means that the type rule call handler (or another one) may be used again
        against the invoked method (__eq__ in our example).

        :param localization: Caller information
        :param rule_params: Rule file entry
        :param arguments: Arguments passed to the call that matches against the rule file.
        :return:
        """
        temp_rule = []
        needs_reevaluation = False
        for i in range(len(rule_params)):
            # Are we dealing with a dependent type?
            if isinstance(rule_params[i], DependentType):
                # Invoke it with the parameters we described previously
                correct_invokation, equivalent_type = rule_params[i](
                    localization, *self.__get_arguments(arguments, i, rule_params[i].call_arity))

                # Is the invocation correct?
                if not correct_invokation:
                    # No, return that this rule do not really match
                    return False, None, needs_reevaluation, None
                else:
                    # The equivalent type is the one determined by the dependent type rule invocation
                    if equivalent_type is not None:
                        # By convention, if the declared rule result is UndefinedType, the call will be reevaluated
                        # substituting the dependent type in position i with its equivalent_type
                        if rule_params[i].expected_return_type is UndefinedType:
                            needs_reevaluation = True
                            temp_rule.append(equivalent_type)
                        # By convention, if the declared rule result is DynamicType, it is substituted by its equivalent
                        # type. This is the most common case
                        if rule_params[i].expected_return_type is DynamicType:
                            return True, None, needs_reevaluation, equivalent_type
                        # #TO DO: This fails
                        # if rule_params[i].expected_return_type is equivalent_type.get_python_type():
                        #     needs_reevaluation = True
                        #     temp_rule.append(equivalent_type)
                        # else:
                        #     return False, None, needs_reevaluation, None

                        # Some dependent types have a declared fixed return type (not like our previous example, which
                        # has DynamicType instead. In that case, if the dependent type invocation do not return the
                        # expected type, this means that the match is not valid and another rule has to be used to
                        # resolve the call.
                        if rule_params[i].expected_return_type is not equivalent_type.get_python_type():
                            return False, None, needs_reevaluation, None
                    else:
                        temp_rule.append(rule_params[i])
            else:
                temp_rule.append(rule_params[i])
        return True, tuple(temp_rule), needs_reevaluation, None

    # TODO: Remove?
    # def get_rule_files(self, proxy_obj, callable_entity):
    #     """
    #     Obtain the corresponding rule files to the callable entity, using its name and its containers name.
    #     :param proxy_obj: TypeInferenceProxy that holds the callable_entity
    #     :param callable_entity: Python callable_entity
    #     :return:
    #     """
    #     if inspect.ismethod(callable_entity) or inspect.ismethoddescriptor(callable_entity):
    #         parent_type_rule_file, own_type_rule_file = self.__rule_files(proxy_obj.parent_proxy.parent_proxy.name,
    #                                                                       proxy_obj.parent_proxy.name)
    #     else:
    #         parent_type_rule_file, own_type_rule_file = self.__rule_files(proxy_obj.parent_proxy.name, proxy_obj.name)
    #
    #     parent_exist = os.path.isfile(parent_type_rule_file)
    #     own_exist = os.path.isfile(own_type_rule_file)
    #
    #     return parent_exist, own_exist, parent_type_rule_file, own_type_rule_file

    def applies_to(self, proxy_obj, callable_entity):
        """
        This method determines if this call handler is able to respond to a call to callable_entity. The call handler
        respond to any callable code that has a rule file associated. This method search the rule file and, if found,
        loads and caches it for performance reasons. Cache also allows us to not to look for the same file on the
        hard disk over and over, saving much time. callable_entity rule files have priority over the rule files of
        their parent entity should both exist.

        :param proxy_obj: TypeInferenceProxy that hold the callable entity
        :param callable_entity: Callable entity
        :return: bool
        """
        # We have a class, calling a class means instantiating it
        if inspect.isclass(callable_entity):
            cache_name = proxy_obj.name + ".__init__"
        else:
            cache_name = proxy_obj.name

        # No rule file for this callable (from the cache)
        if self.unavailable_type_rule_cache.get(cache_name, False):
            return False

        # There are a rule file for this callable (from the cache)
        if self.type_rule_cache.get(cache_name, False):
            return True

        # There are a rule file for this callable parent entity (from the cache)
        if proxy_obj.parent_proxy is not None:
            if self.type_rule_cache.get(proxy_obj.parent_proxy.name, False):
                return True

        # TODO: Remove?
        # if proxy_obj.name in self.unavailable_type_rule_cache:
        #     return False
        #
        # if proxy_obj.name in self.type_rule_cache:
        #     return True

        # if proxy_obj.parent_proxy is not None:
        #     if proxy_obj.parent_proxy.name in self.type_rule_cache:
        #         return True

        # Obtain available rule files depending on the type of entity that is going to be called
        if inspect.ismethod(callable_entity) or inspect.ismethoddescriptor(callable_entity) or (
                    inspect.isbuiltin(callable_entity) and
                    (inspect.isclass(proxy_obj.parent_proxy.get_python_entity()))):
            try:
                parent_type_rule_file, own_type_rule_file = self.__rule_files(callable_entity.__objclass__.__module__,
                                                                              callable_entity.__objclass__.__name__,
                                                                              )
            except:
                if inspect.ismodule(proxy_obj.parent_proxy.get_python_entity()):
                    parent_type_rule_file, own_type_rule_file = self.__rule_files(
                        proxy_obj.parent_proxy.name,
                        proxy_obj.parent_proxy.name)
                else:
                    parent_type_rule_file, own_type_rule_file = self.__rule_files(
                        proxy_obj.parent_proxy.parent_proxy.name,
                        proxy_obj.parent_proxy.name)
        else:
            parent_type_rule_file, own_type_rule_file = self.__rule_files(proxy_obj.parent_proxy.name, proxy_obj.name)

        # Determine which rule file to use
        parent_exist = os.path.isfile(parent_type_rule_file)
        own_exist = os.path.isfile(own_type_rule_file)
        file_path = ""

        if parent_exist:
            file_path = parent_type_rule_file

        if own_exist:
            file_path = own_type_rule_file

        # Load rule file
        if parent_exist or own_exist:
            dirname = os.path.dirname(file_path)
            file_ = file_path.split('/')[-1][0:-3]

            sys.path.append(dirname)
            module = __import__(file_, globals(), locals())
            entity_name = proxy_obj.name.split('.')[-1]
            try:
                # Is there a rule for the specific entity even if the container of the entity has a rule file?
                # This way rule files are used while they are created. All rule files declare a member called
                # type_rules_of_members
                rules = module.type_rules_of_members[entity_name]

                # Dynamically load-time calculated rules (unused yet)
                if inspect.isfunction(rules):
                    rules = rules()  # rules(entity_name)

                # Cache loaded rules for the member
                self.type_rule_cache[cache_name] = rules
            except:
                # Cache unexisting rules for the member
                self.unavailable_type_rule_cache[cache_name] = True
                return False

        if not (parent_exist or own_exist):
            if proxy_obj.name not in self.unavailable_type_rule_cache:
                # Cache unexisting rules for the member
                self.unavailable_type_rule_cache[cache_name] = True

        return parent_exist or own_exist

    def __get_rules_and_name(self, entity_name, parent_name):
        """
        Obtain a member name and its type rules
        :param entity_name: Entity name
        :param parent_name: Entity container name
        :return: tuple (name, rules tied to this name)
        """
        if entity_name in self.type_rule_cache:
            name = entity_name
            rules = self.type_rule_cache[entity_name]

            return name, rules

        if parent_name in self.type_rule_cache:
            name = parent_name
            rules = self.type_rule_cache[parent_name]

            return name, rules

    @staticmethod
    def __format_admitted_params(name, rules, arguments, call_arity):
        """
        Pretty-print error message when no type rule for the member matches with the arguments of the call
        :param name: Member name
        :param rules: Rules tied to this member name
        :param arguments: Call arguments
        :param call_arity: Call arity
        :return:
        """
        params_strs = [""] * call_arity
        first_rule = True
        arities = []

        # Problem with argument number?
        rules_with_enough_arguments = False
        for (params_in_rules, return_type) in rules:
            rule_len = len(params_in_rules)
            if rule_len not in arities:
                arities.append(rule_len)

            if len(params_in_rules) == call_arity:
                rules_with_enough_arguments = True

        if not rules_with_enough_arguments:
            str_arities = ""
            for i in range(len(arities)):
                str_arities += str(arities[i])
                if len(arities) > 1:
                    if i == len(arities) - 1:
                        str_arities += " or "
                    else:
                        str_arities += ", "
            return "The invocation was performed with {0} argument(s), but only {1} argument(s) are accepted".format(
                call_arity,
                str_arities)

        for (params_in_rules, return_type) in rules:
            if len(params_in_rules) == call_arity:
                for i in range(call_arity):
                    value = str(params_in_rules[i])
                    if value not in params_strs[i]:
                        if not first_rule:
                            params_strs[i] += " \/ "
                        params_strs[i] += value

                first_rule = False

        repr_ = ""
        for str_ in params_strs:
            repr_ += str_ + ", "

        return name + "(" + repr_[:-2] + ") expected"

    @staticmethod
    def __compare(params_in_rules, argument_types):
        """
        Most important function in the call handler, determines if a rule matches with the call arguments initially
        (this means that the rule can potentially match with the argument types because the structure of the arguments,
        but if the rule has dependent types, this match could not be so in the end, once the dependent types are
        evaluated.
        :param params_in_rules: Parameters declared on the rule
        :param argument_types: Types passed on the call
        :return:
        """
        for i in range(len(params_in_rules)):
            param = params_in_rules[i]
            # Always should be declared at the end of the rule list for a member, so no more iterations should occur
            if isinstance(param, VarArgType):
                continue
            # Type group: An special entity that matches against several Python types (it overloads its __eq__ method)
            if isinstance(param, BaseTypeGroup):
                if not param == argument_types[i]:
                    return False
            else:
                # Match against raw Python types
                if not param == argument_types[i].get_python_type():
                    return False

        return True

    @staticmethod
    def __create_return_type(localization, ret_type, argument_types):
        """
        Create a suitable return type for the rule (if the return type is a dependent type, this invoked it against
        the call arguments to obtain it)
        :param localization: Caller information
        :param ret_type: Declared return type in a matched rule
        :param argument_types: Arguments of the call
        :return:
        """
        if isinstance(ret_type, DependentType):
            return type_inference_copy.type_inference_proxy.TypeInferenceProxy.instance(
                ret_type(localization, argument_types))
        else:
            return type_inference_copy.type_inference_proxy.TypeInferenceProxy.instance(
                ret_type)

    def get_parameter_arity(self, proxy_obj, callable_entity):
        """
        Obtain the minimum and maximum arity of a callable element using the type rules declared for it. It also
        indicates if it has varargs (infinite arity)
        :param proxy_obj: TypeInferenceProxy that holds the callable entity
        :param callable_entity: Callable entity
        :return: list of possible arities, bool (wether it has varargs or not)
        """
        if inspect.isclass(callable_entity):
            cache_name = proxy_obj.name + ".__init__"
        else:
            cache_name = proxy_obj.name

        has_varargs = False
        arities = []
        name, rules = self.__get_rules_and_name(cache_name, proxy_obj.parent_proxy.name)
        for (params_in_rules, return_type) in rules:
            if self.__has_varargs_in_rule_params(params_in_rules):
                has_varargs = True
            num = len(params_in_rules)
            if num not in arities:
                arities.append(num)

        return arities, has_varargs

    def __call__(self, proxy_obj, localization, callable_entity, *arg_types, **kwargs_types):
        """
        Calls the callable entity with its type rules to determine its return type.

        :param proxy_obj: TypeInferenceProxy that hold the callable entity
        :param localization: Caller information
        :param callable_entity: Callable entity
        :param arg_types: Arguments
        :param kwargs_types: Keyword arguments
        :return: Return type of the call
        """
        if inspect.isclass(callable_entity):
            cache_name = proxy_obj.name + ".__init__"
        else:
            cache_name = proxy_obj.name

        name, rules = self.__get_rules_and_name(cache_name, proxy_obj.parent_proxy.name)

        argument_types = None

        # If there is only one rule, we transfer to the rule the ability of reporting its errors more precisely
        if len(rules) > 1:
            prints_msg = True
        else:
            prints_msg = False

        # Method?
        if inspect.ismethod(callable_entity) or inspect.ismethoddescriptor(callable_entity):
            # Are we calling with a type variable instead with a type instance?
            if not proxy_obj.parent_proxy.is_type_instance():
                # Is the first parameter a subtype of the type variable used to perform the call?
                if not issubclass(arg_types[0].python_entity, callable_entity.__objclass__):
                    # No: Report a suitable error
                    argument_types = tuple(list(arg_types) + kwargs_types.values())
                    usage_hint = self.__format_admitted_params(name, rules, argument_types, len(argument_types))
                    arg_description = str(argument_types)
                    arg_description = arg_description.replace(",)", ")")
                    return TypeError(localization,
                                     "Call to {0}{1} is invalid. Argument 1 requires a '{3}' but received "
                                     "a '{4}' \n\t{2}".format(name, arg_description, usage_hint,
                                                              str(callable_entity.__objclass__),
                                                              str(arg_types[0].python_entity)),
                                     prints_msg=prints_msg)
                else:
                    argument_types = tuple(list(arg_types[1:]) + kwargs_types.values())

        # Argument types passed for the call (if not previously initialized)
        if argument_types is None:
            argument_types = tuple(list(arg_types))  # + kwargs_types.values())

        call_arity = len(argument_types)

        # Examine each rule corresponding to this member
        for (params_in_rules, return_type) in rules:
            # Discard rules that do not match arity
            if len(params_in_rules) == call_arity or self.__has_varargs_in_rule_params(params_in_rules):
                # The passed arguments matches with one of the rules
                if self.__compare(params_in_rules, argument_types):
                    # Is there a dependent type on the matched rule?
                    if self.__dependent_type_in_rule_params(params_in_rules):
                        # We obtain the equivalent type rule of the matched rule (calculated during comparison)
                        correct, equivalent_rule, needs_reevaluation, invokation_rt = self.invoke_dependent_rules(
                            localization, params_in_rules, argument_types)
                        # Errors in dependent type invocation?
                        if correct:
                            # Correct call, return the rule declared type
                            if not needs_reevaluation:
                                # The rule says that this is the type to be returned, as it couldn't be predetermined
                                # in the rule
                                if invokation_rt is not None:
                                    return self.__create_return_type(localization, invokation_rt, argument_types)
                                    # return type_inference.type_inference_proxy.TypeInferenceProxy.instance(
                                    #     invokation_rt)

                                # Comprobacion de Dependent return type
                                return self.__create_return_type(localization, return_type, argument_types)
                                # return type_inference.type_inference_proxy.TypeInferenceProxy.instance(return_type)
                            else:
                                # As one of the dependent rules has a non-predefined return type, we need to obtain it
                                # and evaluate it again
                                for (params_in_rules2, return_type2) in rules:
                                    # The passed arguments matches with one of the rules
                                    if params_in_rules2 == equivalent_rule:
                                        # Create the return type
                                        return self.__create_return_type(localization, return_type2, argument_types)
                    else:
                        # Create the return type
                        return self.__create_return_type(localization, return_type, argument_types)

        # No rule is matched, return error
        usage_hint = self.__format_admitted_params(name, rules, argument_types, len(argument_types))

        arg_description = str(argument_types)
        arg_description = arg_description.replace(",)", ")")
        return TypeError(localization, "Call to {0}{1} is invalid.\n\t{2}".format(name, arg_description,
                                                                                  usage_hint), prints_msg=prints_msg)
