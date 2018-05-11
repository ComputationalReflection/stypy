import inspect

from ...python_lib_copy.member_call_copy.handlers_copy import type_rule_call_handler_copy
from ...python_lib_copy.member_call_copy.handlers_copy import fake_param_values_call_handler_copy
from ...python_lib_copy.member_call_copy.handlers_copy import user_callables_call_handler_copy
from ...python_lib_copy.member_call_copy.type_modifiers_copy import file_type_modifier_copy
from arguments_unfolding_copy import *
from call_handlers_helper_methods_copy import *

"""
Call handlers are the entities we use to perform calls to type inference code. There are several call handlers, as
the call strategy is different depending on the origin of the code to be called:

- Rule-based call handlers: This is used with Python library modules and functions.
Some of these elements may have a rule file associated. This rule file indicates the accepted
parameters for this call and it expected return type depending on this parameters. This is the most powerful call
handler, as the rules we developed allows a wide range of type checking options that may be used to ensure valid
calls. However, rule files have to be developed for each Python module, and while we plan to develop rule files
for each one of them on a semi-automatic way, this is the last part of the stypy development process, which means
that not every module will have one. If no rule file is present, other call handler will take care of the call.

Type rules are read from a directory structure inside the library, so we can add them on a later stage of development
without changing stypy source code.

- User callables call handler: The existence of a rule-based call handler is justified by the inability to have the
code of Python library functions, as most of them are developed in C and the source code cannot be obtained anyway.
However, user-coded .py files are processed and converted to a type inference equivalent program. The conversion
of callable entities transform them to a callable form composed by two parameters: a list of variable arguments and
a list of keyword arguments (def converted_func(*args, **kwargs)) that are handled by the type inference code. This
call handler is the responsible of passing the parameters in this form, so we can call type inference code easily.

- Fake param values call handler: The last-resort call handler, used in those Python library modules with no current
type rule file and external third-party code that cannot be transformed to type inference code because source code
is not available. Calls to this type of code from type inference code will pass types instead of values to the call.
 For example, if we find in our program the call library_function_with_no_source_code(3, "hi") the type inference
 code we generate will turn this to library_function_with_no_source_code(*[int, str], **{}). As this call is not valid
 (the called function cannot be transformed to a type inference equivalent), this call handler obtains default
 predefined fake values for each passed type and phisically call the function with them in order to obtain a result.
 The type of this result is later returned to the type inference code. This is the functionality of this call handler.
 Note that this dynamically obtain the type of a call by performing the call, causing the execution of part of the
 real program instead of the type-inference equivalent, which is not optimal. However, it allows us to test a much
 wider array of programs initially, even if they use libraries and code that do not have the source available and
 have no type rule file attached to it. It is our goal, with time to rely on this call handler as less as possible.
 Note that if the passed type has an associated value, this value will be used instead of the default fake one. However,
 as we said, type values are only calculated in very limited cases.
"""

# We want the type-rule call handler instance available
rule_based_call_handler = type_rule_call_handler_copy.TypeRuleCallHandler()

"""
Here we register, ordered by priority, those classes that handle member calls using different strategies to obtain
the return type of a callable that we described previously, once the type or the input parameters are obtained. Note
that all call handlers are singletons, stateless classes.
"""
registered_call_handlers = [
    rule_based_call_handler,
    user_callables_call_handler_copy.UserCallablesCallHandler(),
    fake_param_values_call_handler_copy.FakeParamValuesCallHandler(),
]

"""
A type modifier is an special class that is associated with type-rule call handler, complementing its functionality.
Although the rules we developed are able to express the return type of a Python library call function in a lot of
cases, there are cases when they are not enough to accurately express the shape of the return type of a function.
This is true when the return type is a collection of a certain type, for example. This is when a type modifier is
used: once a type rule has been used to determine that the call is valid, a type modifier associated to this call
is later called with the passed parameters to obtain a proper, more accurate return type than the expressed by the rule.
Note that not every Python library callable will have a type modifier associated. In fact most of them will not have
one, as this is only used to improve type inference on certain specific callables, whose rule files are not enough for
that. If a certain callable has both a rule file return type and a type modifier return type, the latter takes
precedence.

Only a type modifier is present at the moment: The one that dynamically reads type modifier functions for a Python
(.py) source file. Type modifiers are read from a directory structure inside the library, so we can add them on a
 later stage of development without changing stypy source code. Although only one type modifier is present, we
 developed this system to add more in the future, should the necessity arise.
"""
registered_type_modifiers = [
    file_type_modifier_copy.FileTypeModifier(),
]


def get_param_arity(proxy_obj, callable_):
    """
    Uses python introspection over the callable element to try to guess how many parameters can be passed to the
    callable. If it is not possible (Python library functions do not have this data), we use the type rule call
    handler to try to obtain them. If all fails, -1 is returned. This function also determines if the callable
    uses a variable list of arguments.
    :param proxy_obj: TypeInferenceProxy representing the callable
    :param callable_: Python callable entity
    :return: list of maximum passable arguments, has varargs tuple
    """
    # Callable entity with metadata
    if hasattr(callable_, "im_func"):
        argspec = inspect.getargspec(callable_)
        real_args = len(
            argspec.args) - 2  # callable_.im_func.func_code.co_argcount - 2 #Do not consider localization and self
        has_varargs = argspec.varargs is not None
        return [real_args], has_varargs
    else:
        if rule_based_call_handler.applies_to(proxy_obj, callable_):
            return rule_based_call_handler.get_parameter_arity(proxy_obj, callable_)

    return [-1], False  # Unknown parameter number


def perform_call(proxy_obj, callable_, localization, *args, **kwargs):
    """
    Perform the type inference of the call to the callable entity, using the passed arguments and a suitable
    call handler to resolve the call (see above).

    :param proxy_obj: TypeInferenceProxy representing the callable
    :param callable_: Python callable entity
    :param localization: Caller information
    :param args: named arguments plus variable list of arguments
    :param kwargs: keyword arguments plus defaults
    :return: The return type of the called element
    """

    # Obtain the type of the arguments as a modifiable list
    arg_types = list(args)
    kwarg_types = kwargs

    # TODO: Remove?
    # arg_types = get_arg_types(args)
    # Obtain the types of the keyword arguments
    # kwarg_types = get_kwarg_types(kwargs)

    # Initialize variables
    unfolded_arg_tuples = None
    return_type = None
    found_valid_call = False
    found_errors = []
    found_type_errors = False

    try:
        # Process call handlers in order
        for call_handler in registered_call_handlers:
            # Use the first call handler that declares that can handle this callable
            if call_handler.applies_to(proxy_obj, callable_):
                # When calling the callable element, the type of some parameters might be undefined (not initialized
                # to any value in the preceding code). This function check this fact and substitute the Undefined
                # parameters by suitable type errors. It also creates warnings if the undefined type is inside a
                # union type, removing the undefined type from the union afterwards.
                arg_types, kwarg_types = check_undefined_type_within_parameters(localization,
                                                                                format_call(callable_, arg_types,
                                                                                            kwarg_types),
                                                                                *arg_types, **kwarg_types)

                # Is this a callable that has been converted to an equivalent type inference function?
                if isinstance(call_handler, user_callables_call_handler_copy.UserCallablesCallHandler):
                    # Invoke the applicable call handler
                    ret = call_handler(proxy_obj, localization, callable_, *arg_types, **kwarg_types)
                    if not isinstance(ret, TypeError):
                        # Not an error? accumulate the return type in a return union type
                        found_valid_call = True
                        return_type = ret
                    else:
                        # Store found errors
                        found_errors.append(ret)
                else:
                    # If we reach this point, it means that we have to use type rules or fake param values call handlers
                    # These call handlers do not handle union types, only concrete ones. Therefore, if the passed
                    # arguments have union types, these arguments have to be unfolded: each union type is separated
                    # into its stored types and a new parameter list is formed with each one of them. For example,
                    # if the parameters are ((int \/ str), float), this is generated:
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
                        unfolded_arg_tuples = unfold_arguments(*arg_types, **kwarg_types)

                    # Use each possible combination of union type components found in args / kwargs
                    for tuple_ in unfolded_arg_tuples:
                        # If this parameter tuple contain a type error, do no type inference with it
                        if exist_a_type_error_within_parameters(localization, *tuple_[0], **tuple_[1]):
                            found_type_errors = True
                            continue

                        # Call the call handler with no union type
                        ret = call_handler(proxy_obj, localization, callable_, *tuple_[0], **tuple_[1])
                        if not isinstance(ret, TypeError):
                            # Not an error? accumulate the return type in a return union type. Call is possible with
                            # at least one combination of parameters.
                            found_valid_call = True

                            # As the call is possible with this parameter tuple, we must check possible type modifiers.
                            # Its return type prevails over the returned by the call handler
                            for modifier in registered_type_modifiers:
                                if modifier.applies_to(proxy_obj, callable_):
                                    if inspect.ismethod(callable_) or inspect.ismethoddescriptor(callable_):
                                        # Are we calling with a type variable instead with a type instance?
                                        if not proxy_obj.parent_proxy.is_type_instance():
                                            # Invoke the modifier with the appropriate parameters
                                            ret = modifier(tuple_[0][0], localization, callable_, *tuple_[0][1:],
                                                           **tuple_[1])
                                            break
                                    # Invoke the modifier with the appropriate parameters
                                    ret = modifier(proxy_obj, localization, callable_, *tuple_[0], **tuple_[1])
                                    break

                            # Add the return type of the type rule or the modifier (if one is available for this call)
                            # to the final return type
                            return_type = union_type_copy.UnionType.add(return_type, ret)
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
                        TypeError.remove_error_msg(error)

                    call_str = format_call(callable_, arg_types, kwarg_types)
                    if found_type_errors:
                        msg = "Type errors found among the types of the call parameters"
                    else:
                        msg = "The called entity do not accept any of these parameters"

                    return TypeError(localization, "{0}: {1}".format(call_str, msg))

    except Exception as e:
        # This may indicate an stypy bug
        return TypeError(localization, "An error was produced when invoking '{0}' with arguments [{1}]: {2}".format(
            callable_, list(arg_types) + list(kwarg_types.values()), e))
