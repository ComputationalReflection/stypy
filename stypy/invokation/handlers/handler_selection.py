#!/usr/bin/env python
# -*- coding: utf-8 -*-
from arguments_unfolding import unfold_arguments
from default_handler import DefaultHandler
from cannot_resolve_handler import CannotResolveTypeHandler
from stypy.errors.type_error import StypyTypeError
from stypy.invokation.type_rules.type_groups.type_groups import DynamicType
from stypy.reporting.output_formatting import format_call
from stypy.types import union_type
from stypy.types.undefined_type import UndefinedType
from type_inference_programs_handler import TypeInferenceProgramsHandler
from type_rules_handler import TypeRulesHandler

default_handler = DefaultHandler() #CannotResolveTypeHandler() # DefaultHandler()
handler_priority = [TypeRulesHandler(), TypeInferenceProgramsHandler(), default_handler]


def __exist_a_type_error_within_parameters(*arg_types, **kwargs_types):
    """
    Is there at least a type error among the call parameters?
    :param arg_types: Call arguments
    :param kwargs_types: Call keyword arguments
    :return: bool
    """
    t_e_args = filter(lambda elem: isinstance(elem, StypyTypeError), arg_types)
    if len(t_e_args) > 0:
        return True

    t_e_kwargs = filter(lambda elem: isinstance(elem, StypyTypeError), kwargs_types.values())
    if len(t_e_kwargs) > 0:
        return True

    return False


def __get_first_type_error_within_parameters(*arg_types, **kwargs_types):
    """
    Is there at least a type error among the call parameters?
    :param arg_types: Call arguments
    :param kwargs_types: Call keyword arguments
    :return: bool
    """
    t_e_args = filter(lambda elem: isinstance(elem, StypyTypeError), arg_types)
    if len(t_e_args) > 0:
        return t_e_args[0]

    t_e_kwargs = filter(lambda elem: isinstance(elem, StypyTypeError), kwargs_types.values())
    if len(t_e_kwargs) > 0:
        return t_e_kwargs[0]

    return None


def invoke(localization, callable_, *args, **kwargs):
    unfolded_arg_tuples = None
    found_errors = list()
    found_valid_call = False
    return_type = None
    found_type_errors = False

    #localization = localization.get_current()
    if isinstance(callable_, union_type.UnionType):
        return callable_.invoke(localization, *args, **kwargs)

    # Calling anything over a DynamicType immediately returns another DynamicType
    if isinstance(callable_, DynamicType):
        return callable_

    for handler in handler_priority:
        handler_data = handler.can_be_applicable_to(callable_)
        if handler_data is not None:
            if handler.supports_union_types():
                return handler(handler_data, localization, callable_, *args, **kwargs)
            else:
                # If we reach this point, it means that we have to use a call handler that cannot process union types,
                # only concrete ones. Therefore, if the passed arguments have union types, these arguments have to be
                # unfolded: each union type is separated into its stored types and a new parameter list is formed with
                # each one of them. For example, if the parameters are ((int \/ str), float), this is generated:
                # [
                #   (int, float),
                #   (str, float)
                # ]
                #
                # This process is repeated with any union type found among parameters on any position, so at the
                # end we have multiple possible parameter lists all with no union types present. Later on, all
                # the possible parameter lists are checked with the call handler, and results of all of them are
                # collected to obtain the final call result.

                if unfolded_arg_tuples is None:
                    # Unfold union types found in arguments and/or keyword arguments to use proper python types
                    # in functions calls.
                    unfolded_arg_tuples = unfold_arguments(*args, **kwargs)

                tuple_ = None
                # Use each possible combination of union type components found in args / kwargs
                for tuple_ in unfolded_arg_tuples:
                    # If this parameter tuple contain a type error, do no type inference with it
                    if __exist_a_type_error_within_parameters(localization, *tuple_[0], **tuple_[1]):
                        found_type_errors = True
                        continue

                    # Call the call handler with no union type
                    temp_undef = filter(lambda p: isinstance(p, UndefinedType) or p is UndefinedType, tuple_[0])

                    undefined_types_in_params = len(temp_undef) > 0
                    ret = handler(handler_data, localization, callable_, *tuple_[0], **tuple_[1])
                    if not isinstance(ret, StypyTypeError):
                        # Not an error? accumulate the return type in a return union type. Call is possible with
                        # at least one combination of parameters.
                        if undefined_types_in_params:
                            call_str = format_call(callable_, args, kwargs)
                            found_errors.append(StypyTypeError(localization,
                                                               "Undefined types found among the parameters of the call: " +
                                                               call_str))
                        else:
                            found_valid_call = True

                        # Add the return type of the type rule or the modifier (if one is available for this call)
                        # to the final return type
                        return_type = union_type.UnionType.add(return_type, ret)
                    else:
                        # Store found errors
                        found_errors.append(ret)

                # Could we found any valid combination? Then return the union type that represent all the possible
                # return types of all the valid parameter combinations and convert the call errors to warnings
                if found_valid_call:
                    for error in found_errors:
                        error.turn_to_warning()
                    return return_type
                else:
                    # No possible combination of parameters is possible? Remove all errors and return a single one
                    # with an appropriate message

                    # Only one error? then return it as the cause
                    if len(found_errors) == 1:
                        return found_errors[0]

                    # Multiple error found? We tried to return a compendium of all the obtained error messages, but
                    # we found that they were not clear or even misleading, as the user don't have to be aware of what
                    # a union type is. So, in the end we decided to remove all the generated errors and return a single
                    # one with a generic descriptive message and a pretty-print of the call.
                    for error in found_errors:
                        StypyTypeError.remove_error_msg(error)

                    call_str = format_call(callable_, args, kwargs)
                    if found_type_errors:
                        return __get_first_type_error_within_parameters(localization, *tuple_[0], **tuple_[1])
                    else:
                        msg = "The called entity do not accept any of these parameters"

                    return StypyTypeError(localization, "{0}: {1}".format(call_str, msg))

    return default_handler(None, localization, callable_, *args, **kwargs)
