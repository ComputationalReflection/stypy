#!/usr/bin/env python
# -*- coding: utf-8 -*-
import instance_to_type
from abstract_call_handler import AbstractCallHandler
from stypy.errors.type_error import StypyTypeError
from stypy.invokation.type_rules.type_groups.type_groups import DynamicType
from stypy.module_imports.python_library_modules import is_python_library_module
from stypy.reporting.output_formatting import format_call
from stypy.types.type_inspection import get_defining_module, is_pyd_module
from stypy.types.undefined_type import UndefinedType
from stypy.types import type_wrapper, union_type

cont = 0


class DefaultHandler(AbstractCallHandler):
    """
    The default call handler is used when a call cannot be analyzed using either type rules or calling the generated
    type inference equivalent program. In this case, a call to the Python code is performed using sample instances of
    the passed parameter types. Once a value is returned, its type is calculated and returned to the caller, as
    call handlers are used within the code of a type inference program.

    Note that this call handler is a last resort measure if none of the others can work with the callable entity.
    Although its success rate is high, it can fail due to the nature of the handler. The passed parameters are sample
    instances of its types (the generated type inference program lose the concrete values of most types , and if a call
    needs, for example, values within a range of values to work properly it will return an error.
    """

    print_rules = False

    def can_be_applicable_to(self, callable_):
        """
        This handler is always applicable. Therefore, it must be examined the last
        :param callable_:
        :return:
        """
        return True

    @staticmethod
    def __is_stypy_module(module_name):
        """
        Determines if a file belongs to the stypy modules
        :param module_name:
        :return:
        """
        return "stypy" in module_name

    def __unwrap(self, obj):
        if isinstance(obj, type_wrapper.TypeWrapper):
            return obj.wrapped_type
        return obj

    def __decouple_union_types(self, obj, args):
        """
        If a union type reaches this point it means that stypy has performed a dynamic invokation of a function. This
        means that union types are containing just the different parameters to be passed to the func, that will be
        decoupled and put into the args list in order. This is not a 100% effective solution. If this do not cover the
        need, type rules should be used.
        :param obj:
        :param args:
        :return:
        """
        if isinstance(obj, union_type.UnionType):
            typs = obj.get_types()
            args.extend(typs)
        else:
            args.append(obj)

        return obj

    def __print_rule(self, module, callable_, passed_params, ret, keywork_args=None):
        if not DefaultHandler.print_rules:
            return

        try:
            if module is None:
                if hasattr(callable_, '__self__'):
                    module = type(callable_.__self__)
                else:
                    pass

            strtip = "\t(("
            for param in passed_params:
                strtip += type(param).__name__ + ", "
            if len(passed_params) > 1:
                strtip = strtip[:-2]

            strtip += "), "
            strtip += type(ret).__name__ + "),"

            print "**** RULE:\n" + str(module) + " -> \n'" + callable_.__name__ + "': [\n" + strtip + "\n]\n"
        except:
            pass

    def __call__(self, applicable_rules, localization, callable_, *arguments, **keyword_arguments):
        """
        Perform the call
        :param applicable_rules: Unused
        :param localization:
        :param callable_:
        :param arguments:
        :param keyword_arguments:
        :return:
        """
        global cont
        passed_params = arguments
        #if len(keyword_arguments) > 0:
            #passed_params = list(passed_params) + [keyword_arguments]

        has_undefined_type_in_arguments = len(
            filter(lambda arg: arg is UndefinedType, arguments)) > 0 or callable_ is UndefinedType
        has_dynamic_type_in_arguments = len(filter(lambda arg: isinstance(arg, DynamicType), arguments)) > 0

        has_undefined_type_in_arguments |= len(filter(lambda arg: arg is UndefinedType, keyword_arguments.values())) > 0
        has_dynamic_type_in_arguments |= len(
            filter(lambda arg: isinstance(arg, DynamicType), keyword_arguments.values())) > 0

        if has_undefined_type_in_arguments:
            return UndefinedType
        if has_dynamic_type_in_arguments:
            return DynamicType()

        if len(keyword_arguments) == 1:
            kwvalues = keyword_arguments.values()
            if len(kwvalues) > 0:
                if isinstance(kwvalues[0], type_wrapper.TypeWrapper):
                    kwvalues = kwvalues[0].get_wrapped_type()
                    kwvalues = kwvalues.values()
                    if len(kwvalues) > 0:
                        if type(kwvalues[0]) is UndefinedType or isinstance(kwvalues[0], UndefinedType):
                            keyword_arguments = dict()
                    else:
                        keyword_arguments = dict()
            else:
                keyword_arguments = dict()
        try:
            module = get_defining_module(callable_)
            if is_pyd_module(module):
                # print ("WARNING: Handling call to a pyd file: " + module + "." + str(callable_))
                passed_params = map(lambda obj: self.__unwrap(obj), passed_params)
                kwtemp = dict()
                for key, value in keyword_arguments.items():
                    kwtemp[key] = self.__unwrap(value)
                keyword_arguments = kwtemp

                real_params = []
                for arg in passed_params:
                    self.__decouple_union_types(arg, real_params)

                passed_params = real_params
                kwtemp = dict()
                for key, value in keyword_arguments.items():
                    kwargs = []
                    self.__decouple_union_types(value, kwargs)
                    if len(kwargs) > 1:
                        kwtemp[key] = kwargs
                    else:
                        kwtemp[key] = value

                cont = 0

                def inc():
                    global cont
                    cont += 1
                    return str(cont)

                argument_names = map(lambda e: "argument" + inc(), arguments)

                localization.set_stack_trace(str(callable_), argument_names, arguments)
                # Call to native code (.pyd)
                if len(keyword_arguments) > 0:
                    ret = callable_(*passed_params, **keyword_arguments)
                    self.__print_rule(module, callable_, passed_params, ret, keyword_arguments)
                else:
                    ret = callable_(*passed_params)
                    self.__print_rule(module, callable_, passed_params, ret)
                localization.unset_stack_trace()

            else:
                # Call to a Python library
                if module is None or is_python_library_module(module) or DefaultHandler.__is_stypy_module(
                        module):
                    #ret = callable_(*passed_params)
                    if module is None:
                        pass
                        # print ("WARNING: Handling call to a None module: " + str(callable_))
                    passed_params = map(lambda obj: self.__unwrap(obj), passed_params)
                    kwtemp = dict()
                    for key, value in keyword_arguments.items():
                        kwtemp[key] = self.__unwrap(value)
                    keyword_arguments = kwtemp

                    real_params = []
                    for arg in passed_params:
                        self.__decouple_union_types(arg, real_params)

                    passed_params = real_params
                    kwtemp = dict()
                    for key, value in keyword_arguments.items():
                        kwargs = []
                        self.__decouple_union_types(value, kwargs)
                        if len(kwargs) > 1:
                            kwtemp[key] = kwargs
                        else:
                            kwtemp[key] = value

                    # cont = 0
                    #
                    # def inc():
                    #     global cont
                    #     cont += 1
                    #     return str(cont)
                    #
                    # argument_names = map(lambda e: "argument" + inc(), arguments)
                    #
                    # localization.set_stack_trace(str(callable_), argument_names, arguments)

                    if len(keyword_arguments) > 0:
                        ret = callable_(*passed_params, **keyword_arguments)
                        self.__print_rule(module, callable_, passed_params, ret, keyword_arguments)
                    else:
                        ret = callable_(*passed_params)
                        self.__print_rule(module, callable_, passed_params, ret)

                    #localization.unset_stack_trace()
                else:
                    # Rest of the calls
                    if len(keyword_arguments) > 0:
                        passed_params = list(passed_params) + [keyword_arguments]
                    ret = callable_(localization, *passed_params)
                    self.__print_rule(module, callable_, passed_params, ret)

        except Exception as ex:
            str_call = format_call(callable_, arguments, keyword_arguments)
            return StypyTypeError(localization, "{0}: {1}".format(str_call, str(ex)))

        return instance_to_type.turn_to_type(ret)
