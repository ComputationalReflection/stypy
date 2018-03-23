#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

from abstract_call_handler import AbstractCallHandler
from stypy.errors.type_error import ConstructorParameterError
from stypy.sgmc.sgmc_main import SGMC
from stypy.types.type_inspection import *


class TypeInferenceProgramsHandler(AbstractCallHandler):
    """
    This invokation handler deals calls to code of type inference programs created from existing python sources
    """

    def supports_union_types(self):
        """
        This handler always supports union types
        :return:
        """
        return True

    @staticmethod
    def __get_module_file(obj):
        """
        Gets the file where the code to be called (obj) is declared
        :param obj:
        :return:
        """
        if inspect.ismodule(obj):
            module = obj
        else:
            try:
                module = sys.modules[get_defining_module(obj)]
            except:
                return None

        if hasattr(module, '__file__'):
            return module.__file__
        else:
            return None

    def can_be_applicable_to(self, callable_):
        """
        Determines if the callable entity can be handler by this handler
        :param callable_:
        :return:
        """
        module_file = TypeInferenceProgramsHandler.__get_module_file(callable_)
        if module_file is None:
            return None

        module_file = module_file.replace(".pyc", ".py")

        # Patch to enable calling union type classes methods
        if "no_recursion.py" in module_file:
            return True

        ti_file = module_file
        if "sgmc_cache" not in ti_file:
            ti_file = SGMC.sgmc_cache_absolute_path + SGMC.get_sgmc_route(module_file)
        if ti_file is None:
            return None
        is_file = os.path.isfile(ti_file)
        if not is_file:
            return None
        return ti_file

    def __call__(self, applicable_rules, localization, callable_python_entity, *arguments, **keyword_arguments):
        """
        Performs the call and return the call result
        :param proxy_obj: TypeInferenceProxy that hold the callable entity
        :param localization: Caller information
        :param callable_entity: Callable entity
        :param arg_types: Arguments
        :param kwargs_types: Keyword arguments
        :return: Return type of the call
        """
        try:
            # Keyword arguments sometimes appear as a dict with an unique temp var name that points to the real
            # keyword dict. This happens when kwargs are assigned from an existing variable / attribute
            if len(keyword_arguments) == 1:
                kwdict = keyword_arguments[keyword_arguments.keys()[0]]
                if isinstance(kwdict, TypeWrapper):
                    if isinstance(kwdict.get_wrapped_type(), dict):
                        keyword_arguments = kwdict

            # Redirect calls to __eq__, __hash__ and other special methods to its type inference equivalents. This is
            # needed to enable stypy to handle the type inference classes properly, as all these methods have special
            # meaning in Python and stypy cannot use its type-inference generated equivalents to its original purpose
            if is_special_name_method(callable_python_entity.__name__):
                method_to_lookup = convert_special_name_method(callable_python_entity.__name__)
                self_obj = get_self(callable_python_entity)
                if hasattr(self_obj, method_to_lookup):
                    return callable_python_entity(localization, *arguments, **keyword_arguments)

            if callable_python_entity.__name__ == "__init__" or inspect.isclass(callable_python_entity):
                constructor_errors_after = filter(lambda err: isinstance(err, ConstructorParameterError),
                                                  StypyTypeError.get_error_msgs())

                if callable_python_entity.__name__ == "__init__" and len(arguments) > 0:
                    # Constructor with parameters
                    # if type(callable_python_entity) == types.MethodType:
                    #     call_result = callable_python_entity(localization, *arguments, **keyword_arguments)
                    # else:
                    call_result = callable_python_entity(arguments[0], localization, *arguments[1:],
                                                         **keyword_arguments)
                else:
                    try:
                        call_result = callable_python_entity(localization, *arguments, **keyword_arguments)
                    except TypeError as ex:
                        call_result = callable_python_entity(*arguments, **keyword_arguments)

                constructor_errors_before = filter(lambda err: isinstance(err, ConstructorParameterError),
                                                   StypyTypeError.get_error_msgs())
                if len(constructor_errors_after) < len(constructor_errors_before):
                    # A constructor error here means that the object instance couldn't be built due to incorrect
                    # parameters passed to the constructor
                    return constructor_errors_before[-1]  # Return last constructor parameter error, if present
            else:
                try:
                    call_result = callable_python_entity(localization, *arguments, **keyword_arguments)
                except TypeError as ex:
                    if "argument" in ex.message:
                        call_result = callable_python_entity(*arguments, **keyword_arguments)
                        if type(call_result) is StypyTypeError:
                            if "Insufficient number of arguments" in call_result.error_msg:
                                if type(callable_python_entity) is types.UnboundMethodType:
                                    try:
                                        error_call_result = call_result
                                        call_result = callable_python_entity(arguments[0], localization, *arguments[:1],
                                                                             **keyword_arguments)
                                        StypyTypeError.remove_error_msg(error_call_result)

                                    except:
                                        raise
                    else:
                        if type(callable_python_entity) is types.UnboundMethodType:
                            try:
                                call_result = callable_python_entity(arguments[0], localization, *arguments[:1], **keyword_arguments)
                            except:
                                raise
                        else:
                            raise
            return call_result
        except Exception as ex:
            return StypyTypeError(localization,
                                  "The attempted call to '{0}' cannot be possible with parameter types {1}: {2}".format(
                                      callable_python_entity, list(arguments) + list(keyword_arguments.values()),
                                      str(ex)))
