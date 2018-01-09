# !/usr/bin/env python
# -*- coding: utf-8 -*-


import copy

from data_in_files_handler import DataInFilesHandler
from stypy import errors
from stypy import stypy_parameters
from stypy import type_inference_programs
from stypy.invokation.type_rules.type_groups.type_groups import VarArgType
from stypy.reporting.output_formatting import format_arguments
from stypy.types.type_inspection import *
from stypy.types.type_wrapper import TypeWrapper
from stypy.types.undefined_type import UndefinedType
from type_rules_dependent_types import *
from type_rules_modifier import TypeRulesModifier


class TypeRulesHandler(DataInFilesHandler):
    """
    This call handler uses type rule files (Python files with a special structure) to determine acceptable parameters
    and return types for the calls of a certain module/class and its callable members. The handler dynamically search,
    load and use these rule files to resolve calls. Optionally, if a type modifiers file is present (a file with code
     associated to the member to execute once a rule is matched) it locates and calls it.
    """

    type_modifier = TypeRulesModifier()

    def __init__(self):
        rule_getter = lambda module, entity_name: module.type_rules_of_members[entity_name]
        super(TypeRulesHandler, self).__init__(stypy_parameters.type_rule_file_postfix, rule_getter,
                                               resolve_function_rules=True)

    @staticmethod
    def __has_varargs_in_rule_params(params):
        """
        Check if a list of params has variable number of arguments
        :param params: List of types
        :return: bool
        """
        return len(filter(lambda par: isinstance(par, VarArgType), params)) > 0

    def __get_parameter_arity_from_rules(self, rules):
        """
        Obtain the minimum and maximum arity of a callable element using the type rules declared for it. It also
        indicates if it has varargs (infinite arity)
        :return: list of possible arities, bool (wether it has varargs or not)
        """

        has_varargs = False
        arities = []
        for (params_in_rules, return_type) in rules:
            if self.__has_varargs_in_rule_params(params_in_rules):
                has_varargs = True
            num = len(params_in_rules)
            if num not in arities:
                arities.append(num)

        return arities, has_varargs

    def get_parameter_arity(self, callable_):
        """
        Uses python introspection over the callable element to try to guess how many parameters can be passed to the
        callable. If it is not possible (Python library functions do not have this data), we use the type rule call
        handler to try to obtain them. If all fails, -1 is returned. This function also determines if the callable
        uses a variable list of arguments.
        :param callable_: Python callable entity
        :return: list of maximum passable arguments, has varargs tuple
        """
        # Callable entity with metadata
        if hasattr(callable_, "im_func"):
            argspec = inspect.getargspec(callable_)
            real_args = len(
                argspec.args) - 2  # Do not consider localization and self
            has_varargs = argspec.varargs is not None
            return [real_args], has_varargs
        else:
            rule_data = self.can_be_applicable_to(callable_)
            if rule_data is not None:
                return self.__get_parameter_arity_from_rules(rule_data)

        return [-1], False  # Unknown parameter number

    @staticmethod
    def get_type_from_type_modifier(localization, callable_entity, parameters):
        """
        If this callable entity has a type modifier, locates and calls it to obtain the final return type.
        :param localization:
        :param callable_entity:
        :param parameters:
        :return:
        """
        handler_data = TypeRulesHandler.type_modifier.can_be_applicable_to(callable_entity)

        if handler_data is not None:
            return TypeRulesHandler.type_modifier(handler_data, localization, callable_entity, *parameters)

        return None

    @staticmethod
    def __get_return_type(localization, calculated_return_type, from_invokation, callable_entity, parameters):
        """
        Gets the return type of a call to an entity
        :param localization:
        :param calculated_return_type:
        :param from_invokation:
        :param callable_entity:
        :param parameters:
        :return:
        """
        if isinstance(calculated_return_type, DependentType):
            return calculated_return_type(localization, parameters)
        else:
            try:
                type_from_modifier = TypeRulesHandler.get_type_from_type_modifier(localization, callable_entity,
                                                                                  parameters)
                if type_from_modifier is not None:
                    if type_from_modifier is types.NoneType:
                        return None

                    return type_from_modifier

                if calculated_return_type is types.NoneType:
                    return None

                if not from_invokation:
                    if calculated_return_type is DynamicType:
                        errors.type_warning.TypeWarning.enable_usage_of_dynamic_types_warning(localization)
                        return DynamicType()
                    builtin_type = type_inference_programs.stypy_interface. \
                        get_builtin_python_type_instance(localization, get_name(calculated_return_type))
                else:
                    builtin_type = calculated_return_type

                if isinstance(builtin_type, StypyTypeError):
                    StypyTypeError.remove_error_msg(builtin_type)
                    return calculated_return_type
                return builtin_type
            except Exception as ex:
                return StypyTypeError(localization,
                                      "Call to '{0}' is invalid.\n\t{1}".format(
                                          get_name(callable_entity),
                                          ex))

    @staticmethod
    def __compatible_arity_func(rule_params, argument_arity):
        """
        Determines if a rule is comparible with a certain parameter arity
        :param rule_params:
        :param argument_arity:
        :return:
        """
        condition_1 = len(rule_params) == argument_arity
        # VarArgs always come at end
        if len(rule_params) > 0:
            condition_2 = isinstance(rule_params[-1], VarArgType) and argument_arity >= len(rule_params[:-1])
        else:
            condition_2 = False

        return condition_1 or condition_2

    @staticmethod
    def __get_type_of(obj):
        """
        Gets the type of an object
        :param obj:
        :return:
        """
        if type(obj) is types.InstanceType:
            return obj

        if obj is types.NoneType:
            return obj

        if obj is UndefinedType:
            return obj

        if isinstance(obj, TypeWrapper):
            return type(obj.wrapped_type)

        return type(obj)

    @staticmethod
    def __get_parameters_and_return_type_from_rule(rule):
        """
        From a rule, obtains its parameters and return type
        :param rule:
        :return:
        """
        rt = rule[1]
        params = list()

        for param in rule[0]:
            if isinstance(param, DependentType):
                params.append(copy.copy(param))
            else:
                params.append(param)

        return params, rt

    def __call__(self, applicable_rules, localization, callable_, *arguments, **keyword_arguments):
        """
        Calls the handler to calculate the return type of the call using the declared type rules and modifiers
        :param applicable_rules:
        :param localization:
        :param callable_:
        :param arguments:
        :param keyword_arguments:
        :return:
        """
        # If keyword arguments have content, we add it to the parameter list
        passed_params = arguments
        from_invokation = False
        if len(keyword_arguments) > 0:
            passed_params = list(passed_params) + [keyword_arguments]

        # passed_params_types = map(lambda param: type(param), passed_params)
        passed_params_types = map(lambda param: TypeRulesHandler.__get_type_of(param), passed_params)
        passed_param_arity = len(passed_params)

        # Look only through rules of the same arity
        # rules follow the format (<parameters>, <return type>)
        same_arity_rules = filter(lambda rule_: self.__compatible_arity_func(rule_[0], passed_param_arity),
                                  applicable_rules)

        has_undefined_type_in_arguments = False
        has_dynamic_type_in_arguments = False

        matches_found = 0

        for rule in same_arity_rules:
            match = True
            parameters, return_type = TypeRulesHandler.__get_parameters_and_return_type_from_rule(rule)
            for i in xrange(passed_param_arity):
                # Undefined type param: we check the rest of the parameters and flag it to return UndefinedType
                if passed_params[i] is UndefinedType:
                    has_undefined_type_in_arguments = True
                    continue

                # Dynamic type param: we check the rest of the parameters and flag it to return DynamicType
                if isinstance(passed_params[i], DynamicType):
                    has_dynamic_type_in_arguments = True
                    continue

                # Variable number of arguments: We allow anything beyond the non-vararg parameters
                if isinstance(parameters[-1], VarArgType) and i >= len(parameters):
                    parameter_at_i = parameters[-1]
                else:
                    parameter_at_i = parameters[i]

                # The parameter (or its type) matches?
                try:
                    if not (parameter_at_i == passed_params_types[i]):
                        match = False
                        break
                    else:
                        if isinstance(parameter_at_i, DependentType):
                            parameter_at_i.set_type(passed_params[i])
                except:
                    if isinstance(parameter_at_i, DependentType):
                        parameter_at_i.set_type(passed_params[i])
                    else:
                        match = False
                        break
            if match:
                matches_found += 1
                if type(return_type) is StypyTypeError:
                    return StypyTypeError(localization, return_type.message)

                if has_undefined_type_in_arguments:
                    return UndefinedType
                if has_dynamic_type_in_arguments:
                    return DynamicType()

                if dependent_type_in_rule_params(parameters):
                    # We obtain the equivalent type rule of the matched rule (calculated during comparison)
                    correct, equivalent_rule, invokation_rt = invoke_dependent_rules(
                        localization, parameters, passed_params)
                    # Errors in dependent type invocation?
                    if correct:
                        # Python could not implement certain operations, but admits its call
                        if type(invokation_rt) is types.NotImplementedType or \
                                        invokation_rt is types.NotImplementedType:
                            continue
                            # not_implemented_error = StypyTypeError(localization,
                            #                       "Python do not implement the call to {0} with parameters {1}".
                            #                       format(get_name(callable_), str(passed_params_types)))

                        # The rule says that this is the type to be returned, as it couldn't be predetermined
                        # in the rule
                        if invokation_rt is not None:
                            return_type = create_return_type(localization, invokation_rt, passed_params_types)
                            from_invokation = True
                        else:
                            return_type = create_return_type(localization, return_type, passed_params_types)
                            from_invokation = False
                    else:
                        continue

                return self.__get_return_type(localization, return_type, from_invokation, callable_, passed_params)

        # No rule was found that potentially matches this call
        if matches_found == 0:
            usage_hint = format_arguments(get_name(callable_), applicable_rules, passed_params, len(passed_params))
            return StypyTypeError(localization, "Call to '{0}' is invalid.\n\t{1}".format(get_name(callable_),
                                                                                          usage_hint))

        reported_errors = filter(lambda err: err.origins_in(localization), StypyTypeError.errors)
        if len(reported_errors) == 0:
            usage_hint = format_arguments(get_name(callable_), applicable_rules, passed_params, len(passed_params))
            return StypyTypeError(localization, "Call to '{0}' is invalid.\n\t{1}".format(get_name(callable_),
                                                                                          usage_hint))
        else:
            return reported_errors[-1]  # Return last reported error in the call
