import inspect
import types

from stypy_copy.errors_copy.type_error_copy import TypeError
from stypy_copy.python_lib_copy.python_types_copy.instantiation_copy.type_instantiation_copy import get_type_sample_value
from stypy_copy.python_lib_copy.python_types_copy.type_copy import Type
from call_handler_copy import CallHandler
from stypy_copy.python_lib_copy.member_call_copy.call_handlers_helper_methods_copy import format_call


class FakeParamValuesCallHandler(CallHandler):
    """
    This handler simulated the call to a callable entity by creating fake values of the passed parameters, actually
    call the callable entity and returned the type of the result. It is used when no other call handler can be
    used to call this entity, meaning that this is third-party code or Python library modules with no source code
    available to be transformed (and therefore using the UserCallablesCallHandler) or this module has no type rule
    file associated (which can be done in the future, and stypy will use the TypeRuleCallHandler instead).
    """

    @staticmethod
    def __get_type_instance(arg):
        """
        Obtain a fake value for the type represented by arg
        :param arg: Type
        :return: Value for that type
        """
        if isinstance(arg, Type):
            # If the TypeInferenceProxy holds an instance, return that instance
            instance = arg.get_instance()
            if instance is not None:
                return instance
            else:
                # If the TypeInferenceProxy holds a value, return that value
                if hasattr(arg, "has_value"):
                    if arg.has_value():
                        return arg.get_value()

                # Else obtain a predefined value for that type
                return get_type_sample_value(arg.get_python_type())

        # Else obtain a predefined value for that type
        return get_type_sample_value(arg)

    @staticmethod
    def __get_arg_sample_values(arg_types):
        """
        Obtain a fake value for all the types passed in a list
        :param arg_types: List of types
        :return: List of values
        """
        return map(lambda arg: FakeParamValuesCallHandler.__get_type_instance(arg), arg_types)

    @staticmethod
    def __get_kwarg_sample_values(kwargs_types):
        """
        Obtain a fake value for all the types passed on a dict. This is used for keyword arguments
        :param kwargs_types: Dict of types
        :return: Dict of values
        """
        kwargs_values = {}
        for value in kwargs_types:
            kwargs_values[value] = FakeParamValuesCallHandler.__get_type_instance(kwargs_types[value])

        return kwargs_values

    def __init__(self, fake_self=None):
        CallHandler.__init__(self)
        self.fake_self = fake_self

    def applies_to(self, proxy_obj, callable_entity):
        """
        This method determines if this call handler is able to respond to a call to callable_entity. The call handler
        respond to any callable code, as it is the last to be used.
        :param proxy_obj: TypeInferenceProxy that hold the callable entity
        :param callable_entity: Callable entity
        :return: Always True
        """
        return True

    def __call__(self, proxy_obj, localization, callable_entity, *arg_types, **kwargs_types):
        """
        This call handler substitutes the param types by fake values and perform a call to the real python callable
        entity, returning the type of the return type if the called entity is not a class (in that case it returns the
        created instance)

        :param proxy_obj: TypeInferenceProxy that hold the callable entity
        :param localization: Caller information
        :param callable_entity: Callable entity
        :param arg_types: Arguments
        :param kwargs_types: Keyword arguments
        :return: Return type of the call
        """

        # Obtain values for all parameters
        arg_values = FakeParamValuesCallHandler.__get_arg_sample_values(arg_types)
        kwargs_values = FakeParamValuesCallHandler.__get_kwarg_sample_values(kwargs_types)

        callable_python_entity = callable_entity
        try:
            if (self.fake_self is not None) and ((not hasattr(callable_python_entity, '__self__')) or (
                        hasattr(callable_python_entity, '__self__') and callable_python_entity.__self__ is None)):
                arg_values = [self.fake_self] + arg_values

            # Call
            call_result = callable_python_entity(*arg_values, **kwargs_values)

            # Calculate the return type
            if call_result is not None:
                if not inspect.isclass(callable_entity):
                    return type(call_result)

                if isinstance(type(call_result).__dict__, types.DictProxyType):
                    if hasattr(call_result, '__dict__'):
                        if not isinstance(call_result.__dict__, types.DictProxyType):
                            return call_result

                    return type(call_result)

            return call_result
        except Exception as ex:
            str_call = format_call(callable_entity, arg_types, kwargs_types)
            return TypeError(localization, "{0}: {1}".format(str_call, str(ex)))
