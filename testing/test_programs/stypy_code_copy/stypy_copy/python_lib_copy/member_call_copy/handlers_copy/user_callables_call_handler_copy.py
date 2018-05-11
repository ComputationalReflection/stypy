import inspect

from ....errors_copy.type_error_copy import TypeError
from call_handler_copy import CallHandler
from ....python_lib_copy.python_types_copy.type_inference_copy import no_recursion_copy
from ....python_lib_copy.python_types_copy.type_inference_copy.type_inference_proxy_management_copy import is_user_defined_class, \
    is_user_defined_module
from ....python_lib_copy.python_types_copy import type_inference_copy


class UserCallablesCallHandler(CallHandler):
    """
    This class handles calls to code that have been obtained from available .py source files that the program uses,
    generated to perform type inference of the original callable code. Remember that all functions in type inference
    programs are transformed to the following form:

    def function_name(*args, **kwargs):
        ...

    This handler just use this convention to call the code and return the result. Transformed code can handle
     UnionTypes and other stypy constructs.
    """

    def applies_to(self, proxy_obj, callable_entity):
        """
        This method determines if this call handler is able to respond to a call to callable_entity.
        :param proxy_obj: TypeInferenceProxy that hold the callable entity
        :param callable_entity: Callable entity
        :return: bool
        """
        return is_user_defined_class(callable_entity) or proxy_obj.parent_proxy.python_entity == no_recursion_copy or \
               (inspect.ismethod(callable_entity) and is_user_defined_class(proxy_obj.parent_proxy.python_entity)) or \
               (inspect.isfunction(callable_entity) and is_user_defined_class(proxy_obj.parent_proxy.python_entity)) or \
               (inspect.isfunction(callable_entity) and is_user_defined_module(callable_entity.__module__))

    def __call__(self, proxy_obj, localization, callable_entity, *arg_types, **kwargs_types):
        """
        Perform the call and return the call result
        :param proxy_obj: TypeInferenceProxy that hold the callable entity
        :param localization: Caller information
        :param callable_entity: Callable entity
        :param arg_types: Arguments
        :param kwargs_types: Keyword arguments
        :return: Return type of the call
        """
        callable_python_entity = callable_entity
        try:
            pre_errors = TypeError.get_error_msgs()
            # None localizations are used to indicate that the callable entity do not use this parameter
            if localization is None:
                call_result = callable_python_entity(*arg_types, **kwargs_types)
            else:
                # static method
                if inspect.isfunction(proxy_obj.python_entity) and inspect.isclass(
                        proxy_obj.parent_proxy.python_entity):
                    call_result = callable_python_entity(proxy_obj.parent_proxy.python_entity,
                                                         localization, *arg_types, **kwargs_types)
                else:
                    # Explicit call to __init__ functions (super)
                    if inspect.ismethod(proxy_obj.python_entity) and ".__init__" in proxy_obj.name and inspect.isclass(
                        proxy_obj.parent_proxy.python_entity):
                        call_result = callable_python_entity(arg_types[0].python_entity,
                                     localization, *arg_types, **kwargs_types)
                    else:
                        # instance method
                        call_result = callable_python_entity(localization, *arg_types, **kwargs_types)

            # Return type
            if call_result is not None:
                if not inspect.isclass(callable_entity):
                    return call_result
                else:
                    # Are we creating an instance of a classs?
                    if isinstance(call_result, callable_entity):
                        post_errors = TypeError.get_error_msgs()
                        if not len(pre_errors) == len(post_errors):
                            return TypeError(localization, "Could not create an instance of ({0}) due to constructor"
                                                           " call errors".format(callable_entity), False)

                    instance = call_result
                    type_ = callable_entity
                    return type_inference_copy.type_inference_proxy.TypeInferenceProxy.instance(type_, instance=instance)
            return call_result
        except Exception as ex:
            return TypeError(localization,
                             "The attempted call to '{0}' cannot be possible with parameter types {1}: {2}".format(
                                 callable_entity, list(arg_types) + list(kwargs_types.values()), str(ex)))
