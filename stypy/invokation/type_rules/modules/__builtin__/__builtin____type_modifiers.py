#!/usr/bin/env python
# -*- coding: utf-8 -*-
import types

from stypy import contexts
from stypy.errors.advice import Advice
from stypy.errors.type_error import StypyTypeError
from stypy.errors.type_warning import TypeWarning
from stypy.invokation.handlers.type_rules_handler import TypeRulesHandler
from stypy.invokation.type_rules.type_groups import type_group_generator
from stypy.module_imports.python_imports import import_from_module
from stypy.module_imports.python_library_modules import is_python_library_module
from stypy.type_inference_programs.stypy_interface import get_builtin_python_type_instance, invoke, python_operator
from stypy.types import union_type
from stypy.types.standard_wrapper import StandardWrapper
from stypy.types.type_containers import get_contained_elements_type, set_contained_elements_type, \
    set_contained_elements_type_for_key, can_store_keypairs, can_store_elements, get_key_types
from stypy.types.type_inspection import is_error, is_str, is_function, is_method, dir_object, is_undefined, compare_type
from stypy.types.type_intercession import get_member, set_member, has_member, del_member
from stypy.reporting.localization import Localization

class TypeModifiers:
    @staticmethod
    def has_member(localization, proxy, member):
        r = get_member(localization, proxy, member)
        StypyTypeError.remove_error_msg(r)

        return not is_error(r)

    @staticmethod
    def type_conversion_check(localization, proxy_obj, func_name, test_func):
        if not hasattr(proxy_obj, '__{0}__'.format(func_name)):
            return StypyTypeError.member_not_defined_error(localization, proxy_obj, "__" + func_name + "__")
        else:
            conversor_call = getattr(proxy_obj, '__{0}__'.format(func_name))
            call_res = invoke(localization, conversor_call)
            if is_error(call_res):
                return call_res
            if not test_func(call_res):
                return StypyTypeError.wrong_return_type_error(localization, "__" + func_name + "__", call_res)

        return call_res

    @staticmethod
    def bytearray(localization, proxy_obj, arguments):
        if len(arguments) == 1:
            ret = get_builtin_python_type_instance(localization, 'bytearray', int(1))
        else:
            ret = get_builtin_python_type_instance(localization, 'bytearray')

        return ret

    @staticmethod
    def set(localization, proxy_obj, arguments):
        ret = get_builtin_python_type_instance(localization, 'set')
        if len(arguments) == 0:
            return ret

        param = arguments[0]
        if is_str(type(param)):
            set_contained_elements_type(ret,
                                        get_builtin_python_type_instance(localization, type(param).__name__))
        else:
            set_contained_elements_type(ret, get_contained_elements_type(param))

        return ret

    @staticmethod
    def vars(localization, proxy_obj, arguments):
        if len(arguments) == 0:
            return TypeModifiers.locals(localization, proxy_obj, arguments)
        else:
            dict_member = get_member(localization, arguments[0], "__dict__")
            if is_error(dict_member):
                return StypyTypeError.object_must_define_member_error(localization, "vars parameter", "__dict__")
            ret = get_builtin_python_type_instance(localization, "dict")

            for key, value in dict_member.iteritems():
                key_type = get_builtin_python_type_instance(localization, type(key).__name__, value=key)
                value_type = value

                set_contained_elements_type_for_key(ret, key_type, value_type)
            return ret

    @staticmethod
    def classmethod(localization, proxy_obj, arguments):
        return classmethod(arguments[0])

    @staticmethod
    def float(localization, proxy_obj, arguments):
        if len(arguments) == 0:
            return get_builtin_python_type_instance(localization, 'float')

        if type_group_generator.Number == type(arguments[0]):
            return get_builtin_python_type_instance(localization, 'float')

        if type_group_generator.Str == type(arguments[0]):
            try:
                if len(arguments[0]) == 0:
                    return float()
                ret = float(arguments[0])

                return float()
            except:
                return StypyTypeError(localization, "Could not convert string to float: '{0}'".format(arguments[0]))

        if TypeModifiers.has_member(localization, arguments[0], "__float__"):
            return TypeModifiers.type_conversion_check(localization, arguments[0], 'float', lambda
                call_result: type(call_result).__name__ == 'float')
        else:
            return get_builtin_python_type_instance(localization, 'float')

    @staticmethod
    def unicode(localization, proxy_obj, arguments):
        if len(arguments) == 0:
            return get_builtin_python_type_instance(localization, 'unicode')

        if TypeModifiers.has_member(localization, arguments[0], "__str__"):
            ret = TypeModifiers.type_conversion_check(localization, arguments[0], 'str', lambda
                call_result: type_group_generator.Str == type(call_result))
            if not is_error(ret):
                return get_builtin_python_type_instance(localization, 'unicode')
            return ret
        else:
            if TypeModifiers.has_member(localization, arguments[0], "stypy__str__"):
                ret = invoke(localization, get_member(localization, arguments[0], "stypy__str__"))
                if not is_error(ret):
                    if type(ret) is not str:
                        return StypyTypeError(localization,
                                              "coercing to Unicode: need string or buffer, {0} found".format(ret))
                    return get_builtin_python_type_instance(localization, 'unicode')
                return ret
            return get_builtin_python_type_instance(localization, 'unicode')

    @staticmethod
    def enumerate(localization, proxy_obj, arguments):
        param = arguments[0]

        if is_str(type(param)):
            enumerate_ = get_builtin_python_type_instance(localization, 'enumerate', ' ')

            tuple_contents = union_type.UnionType.add(0, enumerate_.next()[1])
            contents = get_builtin_python_type_instance(localization, "tuple")
            set_contained_elements_type(contents, tuple_contents)
            set_contained_elements_type(enumerate_, contents)

            return enumerate_
        else:
            if TypeModifiers.has_member(localization, arguments[0], "__iter__"):
                iter_result = TypeModifiers.type_conversion_check(localization, arguments[0], 'iter', lambda
                    call_result: "iterator" in type(call_result.get_wrapped_type()).__name__)

                tuple_contents = union_type.UnionType.add(0, get_contained_elements_type(iter_result))
                contents = get_builtin_python_type_instance(localization, "tuple")
                set_contained_elements_type(contents, tuple_contents)

                enumerate_ = get_builtin_python_type_instance(localization, 'enumerate', ' ')
                set_contained_elements_type(enumerate_, contents)
                return enumerate_
            else:
                contents = union_type.UnionType.add(0, get_contained_elements_type(param))
                enumerate_ = get_builtin_python_type_instance(localization, 'enumerate', ' ')
                set_contained_elements_type(enumerate_, contents)
                return enumerate_

    @staticmethod
    def reduce(localization, proxy_obj, arguments):
        func = arguments[0]

        if is_function(arguments[0]) or is_method(arguments[0]):
            func = arguments[0]
        else:
            func = get_member(localization, arguments[0], "__call__")
            if is_error(func):
                return func

        sequence = arguments[1]
        if len(arguments) > 2:
            initial = arguments[2]
        else:
            initial = None

        if not is_str(type(sequence)):
            contained = get_contained_elements_type(sequence)
        else:
            contained = get_builtin_python_type_instance(localization, type(sequence).__name__)
        try:
            if initial is None:
                ret = invoke(localization, func, contained, contained)
                if is_error(ret):
                    return ret
            else:
                ret = invoke(localization, func, initial, initial)
                if is_error(ret):
                    return ret

            if isinstance(contained, union_type.UnionType):
                contained = contained.types
            else:
                contained = [contained]

            for type_ in contained:
                ret = union_type.UnionType.add(ret, invoke(localization, func, ret, type_))
                if is_error(ret):
                    return ret
        except Exception as ex:
            return StypyTypeError.invalid_callable_error(localization, "callable entity", "reduce function", str(ex))

        return ret

    @staticmethod
    def list(localization, proxy_obj, arguments):
        ret_type = get_builtin_python_type_instance(localization, 'list')
        if len(arguments) > 0:
            params = arguments[0]
            if is_str(type(params)):
                set_contained_elements_type(ret_type,
                                            get_builtin_python_type_instance(localization,
                                                                             type(params).__name__))
            else:
                existing_type = get_contained_elements_type(params)
                if existing_type is not None:
                    set_contained_elements_type(ret_type, existing_type)

        return ret_type

    @staticmethod
    def coerce(localization, proxy_obj, arguments):
        contents = None
        contents = union_type.UnionType.add(contents, arguments[0])
        contents = union_type.UnionType.add(contents, arguments[1])

        tup = get_builtin_python_type_instance(localization, "tuple")
        set_contained_elements_type(tup, contents)

        return tup

    @staticmethod
    def compile(localization, proxy_obj, arguments):
        try:
            return types.CodeType(*arguments)
        except:
            TypeWarning.enable_usage_of_dynamic_types_warning(localization, "compile")
            return type_group_generator.DynamicType()

    @staticmethod
    def globals(localization, proxy_obj, arguments):
        all_globals = globals()
        predefined_globals = {
            '__builtins__': __builtins__,
            '__name__': get_builtin_python_type_instance(localization,
                                                         type(all_globals['__name__']).__name__),
            '__doc__': get_builtin_python_type_instance(localization,
                                                        type(all_globals['__doc__']).__name__),
            '__package__': get_builtin_python_type_instance(localization,
                                                            type(all_globals['__package__']).__name__),
        }
        ret = get_builtin_python_type_instance(localization, 'dict')

        ts = contexts.context.Context.get_current_active_context_for_module(localization.file_name)
        context = ts.get_global_context()

        for key, value in context.types_of.iteritems():
            key_type = get_builtin_python_type_instance(localization, type(key).__name__, value=key)
            value_type = get_builtin_python_type_instance(localization, type(value).__name__, value=value)

            set_contained_elements_type_for_key(ret, key_type, value_type)

        for key, value in predefined_globals.iteritems():
            key_type = get_builtin_python_type_instance(localization, type(key).__name__, value=key)
            value_type = get_builtin_python_type_instance(localization, type(value).__name__)

            set_contained_elements_type_for_key(ret, key_type, value_type)

        return ret

    @staticmethod
    def issubclass(localization, proxy_obj, arguments):
        return get_builtin_python_type_instance(localization, "bool")

    @staticmethod
    def divmod(localization, proxy_obj, arguments):
        # ((x-x%y)/y,x%y)
        x = arguments[0]
        y = arguments[1]
        if not type_group_generator.Number == type(x) and has_member(localization, x, '__divmod__'):
            divmod_func = get_member(localization, x, '__divmod__')
            return invoke(localization, divmod_func, y)

        if not type_group_generator.Number == type(y) and has_member(localization, y, '__rdivmod__'):
            divmod_func = get_member(localization, y, '__rdivmod__')
            return invoke(localization, divmod_func, x)

        mod_res = python_operator(localization, '%', x, y)
        sub_res = python_operator(localization, '-', x, mod_res)
        first_term = python_operator(localization, 'div', sub_res, y)

        if type(first_term) == type(mod_res):
            tup = get_builtin_python_type_instance(localization, "tuple")
            set_contained_elements_type(tup, get_builtin_python_type_instance(localization, type(first_term).__name__))

            return tup

        first_term = get_builtin_python_type_instance(localization, type(first_term).__name__)
        mod_res = get_builtin_python_type_instance(localization, type(mod_res).__name__)

        contents = union_type.UnionType.add(first_term, mod_res)

        tup = get_builtin_python_type_instance(localization, "tuple")
        set_contained_elements_type(tup, contents)

        return tup

    @staticmethod
    def locals(localization, proxy_obj, arguments):
        ret = get_builtin_python_type_instance(localization, 'dict')

        context = contexts.context.Context.get_current_active_context_for_module(localization.file_name)

        for key, value in context.types_of.iteritems():
            key_type = get_builtin_python_type_instance(localization, type(key).__name__, value=key)
            value_type = get_builtin_python_type_instance(localization, type(value).__name__)

            set_contained_elements_type_for_key(ret, key_type, value_type)

        return ret

    @staticmethod
    def unichr(localization, proxy_obj, arguments):
        if type_group_generator.Number == arguments[0]:
            return get_builtin_python_type_instance(localization, 'unicode')

        if TypeModifiers.has_member(localization, arguments[0], '__trunc__'):
            ret = TypeModifiers.type_conversion_check(localization, arguments[0], 'trunc', lambda
                call_result: type(call_result) == int)
            if is_error(ret):
                return ret
            else:
                return get_builtin_python_type_instance(localization, 'unicode')
        else:
            return get_builtin_python_type_instance(localization, 'unicode')

    @staticmethod
    def apply(localization, proxy_obj, arguments):
        if is_function(arguments[0]) or is_method(arguments[0]):
            call_method = arguments[0]
        else:
            call_method = get_member(localization, arguments[0], '__call__')

        if is_error(call_method):
            return call_method

        arities, varargs = TypeRulesHandler().get_parameter_arity(call_method)

        num_params = arities[-1]
        # If we cannot guess the number of parameters call is impossible, so we cannot check it
        if num_params == -1:
            TypeWarning.enable_usage_of_dynamic_types_warning(localization, "apply")
            return type_group_generator.DynamicType()

        if varargs:
            # The last param is a tuple
            num_params -= 1

        param_list = []
        # Build each param with the type of the contained elements of the tuple
        for i in xrange(num_params):
            param_list.append(get_contained_elements_type(arguments[1]))

        if varargs:
            param_list.append(arguments[1])

        if len(arguments) < 3:
            return invoke(localization, call_method, *param_list)
        else:
            param_list.append(arguments[2])
            return invoke(localization, call_method, *param_list)

    @staticmethod
    def file(localization, proxy_obj, arguments):
        file_object = get_builtin_python_type_instance(localization, "file")
        return file_object

    @staticmethod
    def filter(localization, proxy_obj, arguments):
        if arguments[0] is types.NoneType:
            return str()

        if is_function(arguments[0]) or is_method(arguments[0]):
            func = arguments[0]
        else:
            func = get_member(localization, arguments[0], "__call__")
            if is_error(func):
                return func

        if can_store_elements(arguments[1]) or can_store_keypairs(arguments[1]):
            if can_store_keypairs(arguments[1]):
                filter_types = get_key_types(arguments[1])
            else:
                filter_types = get_contained_elements_type(arguments[1])

            resul = invoke(localization, func, filter_types)

            if is_error(resul):
                return resul

            if type(arguments[1]) is tuple:
                ret = get_builtin_python_type_instance(localization, 'tuple')
            else:
                ret = get_builtin_python_type_instance(localization, 'list')

            container_type = filter_types
            set_contained_elements_type(ret, container_type)
        else:
            # For strings
            ret = get_builtin_python_type_instance(localization, 'list')
            container_type = get_builtin_python_type_instance(localization, type(arguments[1]).__name__)
            resul = invoke(localization, func, container_type)
            if is_error(resul):
                return resul
            # container_type = union_type.UnionType.add(container_type, types.NoneType)
            set_contained_elements_type(ret, container_type)

        return ret

    @staticmethod
    def slice(localization, proxy_obj, arguments):
        ret = get_builtin_python_type_instance(localization, 'slice')
        contained = None
        if len(arguments) >= 1:
            if is_function(arguments[0]) or is_method(arguments[0]):
                argument_0 = invoke(localization, arguments[0])
            else:
                callable_ = get_member(localization, arguments[0], '__call__')
                if is_error(callable_):
                    StypyTypeError.remove_error_msg(callable_)
                    argument_0 = arguments[0]
                else:
                    if arguments[0] is not types.NoneType:
                        ret = invoke(localization, callable_)
                        if is_error(ret):
                            return ret
                        argument_0 = ret
                    else:
                        argument_0 = types.NoneType

            contained = union_type.UnionType.add(contained, argument_0)
            if len(arguments) == 1:
                contained = union_type.UnionType.add(contained, types.NoneType)

        if len(arguments) >= 2:
            if is_function(arguments[1]) or is_method(arguments[1]):
                argument_1 = invoke(localization, arguments[1])
            else:
                callable_ = get_member(localization, arguments[1], '__call__')
                if is_error(callable_):
                    StypyTypeError.remove_error_msg(callable_)
                    argument_1 = arguments[1]
                else:
                    if arguments[1] is not types.NoneType:
                        ret = invoke(localization, callable_)
                        if is_error(ret):
                            return ret
                        argument_1 = ret
                    else:
                        argument_1 = types.NoneType

            contained = union_type.UnionType.add(contained, argument_1)

        if len(arguments) == 3:
            if is_function(arguments[2]) or is_method(arguments[2]):
                argument_2 = invoke(localization, arguments[2])
            else:
                callable_ = get_member(localization, arguments[2], '__call__')
                if is_error(callable_):
                    StypyTypeError.remove_error_msg(callable_)
                    argument_2 = arguments[2]
                else:
                    if arguments[2] is not types.NoneType:
                        ret = invoke(localization, callable_)
                        if is_error(ret):
                            return ret
                        argument_2 = ret
                    else:
                        argument_2 = types.NoneType

            contained = union_type.UnionType.add(contained, argument_2)

        set_contained_elements_type(ret, contained)
        return ret

    @staticmethod
    def min(localization, callable_, arguments):
        if type(arguments[-1]) is dict and 'key' in arguments[-1]:
            has_func = True
        else:
            has_func = False

        if can_store_elements(arguments[0]):
            ret_type = get_contained_elements_type(arguments[0])
        else:
            if is_str(type(arguments[0])):
                ret_type = get_builtin_python_type_instance(localization, "str")
            else:
                cont = 0
                ret_type = None
                for arg in arguments:
                    arg_ptype = type(arg)
                    if (not arg_ptype is types.FunctionType and not arg_ptype is types.MethodType
                        and not arg_ptype is types.LambdaType) \
                            or cont < len(arguments) - 1:
                        ret_type = union_type.UnionType.add(ret_type, arg)
                    cont += 1

        if has_func:
            func = arguments[-1]['key']
            for arg in arguments[:-1]:
                if can_store_elements(arg):
                    contained = get_contained_elements_type(arg)
                    result = invoke(localization, func, contained)
                else:
                    result = invoke(localization, func, arg)
                if is_error(result):
                    return result

        return ret_type

    @staticmethod
    def max(localization, proxy_obj, arguments):
        return TypeModifiers.min(localization, proxy_obj, arguments)

    @staticmethod
    def sum(localization, proxy_obj, arguments):
        list_elements = get_contained_elements_type(arguments[0])
        if not isinstance(list_elements, union_type.UnionType):
            list_elements = [list_elements]
        else:
            list_elements = list_elements.types

        ret = None
        for elem in list_elements:
            if is_undefined(elem):
                TypeWarning(localization, "Elements of {0} could be undefined types. Sum may be invalid.".format(type(arguments[0])))
                continue

            if ret is None:
                ret = elem

            add_call = get_member(localization, elem, '__add__')
            if not is_error(add_call):
                ret = python_operator(localization, '+', ret, elem)
            else:
                ret = invoke(localization, add_call, ret, elem)

        return ret

    @staticmethod
    def chr(localization, proxy_obj, arguments):
        if type_group_generator.Integer == arguments[0]:
            return get_builtin_python_type_instance(localization, 'str')
        else:
            trunc_call = get_member(localization, arguments[0], '__trunc__')
            if is_error(trunc_call):
                return trunc_call
            call_res = invoke(localization, trunc_call)
            if is_error(call_res):
                return call_res
            if type(call_res) is not int:
                return StypyTypeError.wrong_return_type_error(localization,
                                                              "__trunc__", type(call_res), "int")

        return get_builtin_python_type_instance(localization, 'str')

    @staticmethod
    def hex(localization, proxy_obj, arguments):
        if type_group_generator.Integer == arguments[0]:
            return get_builtin_python_type_instance(localization, 'str')
        else:
            hex_call = get_member(localization, arguments[0], '__hex__')
            if is_error(hex_call):
                return hex_call
            call_res = invoke(localization, hex_call)
            if is_error(call_res):
                return call_res
            if type(call_res) is not str:
                return StypyTypeError.wrong_return_type_error(localization, '__hex__', type(call_res), 'str')

        return get_builtin_python_type_instance(localization, 'str')

    @staticmethod
    def long(localization, proxy_obj, arguments):
        if len(arguments) == 0:
            return get_builtin_python_type_instance(localization, 'long')

        if type_group_generator.Str == type(arguments[0]):
            try:
                if len(arguments[0]) == 0:
                    return long()
                long(arguments[0])

                return long()
            except:
                return StypyTypeError(localization, "Invalid literal for long(): '{0}'".format(arguments[0]))

        first_is_trunc = False
        second_is_trunc = False

        first_is_int = False
        second_is_int = False

        first_is_long = False
        second_is_long = False

        if len(arguments) > 0:
            first_is_trunc = TypeModifiers.has_member(localization, arguments[0], '__trunc__') and \
                             not type_group_generator.RealNumber == type(arguments[0])

            first_is_int = TypeModifiers.has_member(localization, arguments[0], '__int__') and \
                           not type_group_generator.RealNumber == type(arguments[0])

            first_is_long = TypeModifiers.has_member(localization, arguments[0], '__long__') and \
                            not type_group_generator.RealNumber == type(arguments[0])

        if len(arguments) > 1:
            second_is_trunc = TypeModifiers.has_member(localization, arguments[1], '__trunc__') and \
                              not type_group_generator.Integer == type(arguments[1])

            second_is_int = TypeModifiers.has_member(localization, arguments[1], '__int__') and \
                            not type_group_generator.Integer == type(arguments[1])

            second_is_long = TypeModifiers.has_member(localization, arguments[1], '__long__') and \
                             not type_group_generator.Integer == type(arguments[1])

        if first_is_trunc:
            trunc_res = TypeModifiers.type_conversion_check(localization, arguments[0], 'trunc', lambda
                call_result: type(call_result).__name__ == 'int')
            if is_error(trunc_res):
                return trunc_res

        if second_is_trunc:
            if type_group_generator.RealNumber == type(arguments[1]):
                return StypyTypeError.wrong_parameter_type_error(localization, "Integer", type(arguments[1]).__name__)
            trunc_res = TypeModifiers.type_conversion_check(localization, arguments[1], 'trunc', lambda
                call_result: type(call_result).__name__ == 'int')
            if is_error(trunc_res):
                return trunc_res

        if first_is_int:
            trunc_res = TypeModifiers.type_conversion_check(localization, arguments[0], 'int', lambda
                call_result: type(call_result).__name__ == 'int')
            if is_error(trunc_res):
                return trunc_res

        if second_is_int:
            if type_group_generator.RealNumber == type(arguments[1]):
                return StypyTypeError.wrong_parameter_type_error(localization, "Integer", type(arguments[1]).__name__)
            trunc_res = TypeModifiers.type_conversion_check(localization, arguments[1], 'int', lambda
                call_result: type(call_result).__name__ == 'int')
            if is_error(trunc_res):
                return trunc_res

        if first_is_long:
            trunc_res = TypeModifiers.type_conversion_check(localization, arguments[0], 'long', lambda
                call_result: type(call_result).__name__ == 'long' or type(call_result).__name__ == 'int')
            if is_error(trunc_res):
                return trunc_res

        if second_is_long:
            if type_group_generator.RealNumber == type(arguments[1]):
                return StypyTypeError.wrong_parameter_type_error(localization, "Integer", type(arguments[1]).__name__)
            trunc_res = TypeModifiers.type_conversion_check(localization, arguments[1], 'long', lambda
                call_result: type(call_result).__name__ == 'long' or type(call_result).__name__ == 'int')

            if is_error(trunc_res):
                return trunc_res
        return get_builtin_python_type_instance(localization, 'long')

    @staticmethod
    def xrange(localization, proxy_obj, arguments):
        first_is_trunc = False
        second_is_trunc = False
        third_is_trunc = False

        first_is_int = False
        second_is_int = False
        third_is_int = False

        if len(arguments) > 0:
            if not type_group_generator.Integer == arguments[0]:
                first_is_trunc = TypeModifiers.has_member(localization, arguments[0], '__trunc__') and \
                                 not type_group_generator.RealNumber == type(arguments[0])

                first_is_int = TypeModifiers.has_member(localization, arguments[0], '__int__') and \
                               not type_group_generator.RealNumber == type(arguments[0])

            if first_is_trunc:
                trunc_res = TypeModifiers.type_conversion_check(localization, arguments[0], 'trunc', lambda
                    call_result: type(call_result).__name__ == 'int')
                if is_error(trunc_res):
                    return trunc_res

            if first_is_int:
                trunc_res = TypeModifiers.type_conversion_check(localization, arguments[0], 'int', lambda
                    call_result: type(call_result).__name__ == 'int')
                if is_error(trunc_res):
                    return trunc_res

            if not (first_is_trunc or first_is_int) and not type_group_generator.Integer == type(arguments[0]):
                return StypyTypeError.object_must_be_type_error(localization,
                                                                "'range' first argument", "integer", type(arguments[0]))

        if len(arguments) > 1:
            if not type_group_generator.Integer == arguments[1]:
                second_is_trunc = TypeModifiers.has_member(localization, arguments[1], '__trunc__') and \
                                  not type_group_generator.RealNumber == type(arguments[1])

            if not type_group_generator.Integer == arguments[1]:
                second_is_int = TypeModifiers.has_member(localization, arguments[1], '__int__') and \
                                not type_group_generator.RealNumber == type(arguments[1])

            if second_is_trunc:
                trunc_res = TypeModifiers.type_conversion_check(localization, arguments[1], 'trunc', lambda
                    call_result: type(call_result).__name__ == 'int')
                if is_error(trunc_res):
                    return trunc_res

            if second_is_int:
                trunc_res = TypeModifiers.type_conversion_check(localization, arguments[1], 'int', lambda
                    call_result: type(call_result).__name__ == 'int')
                if is_error(trunc_res):
                    return trunc_res

            if not (second_is_trunc or second_is_int) and not type_group_generator.Integer == type(arguments[1]):
                return StypyTypeError.object_must_be_type_error(localization,
                                                                "'range' second argument", "integer",
                                                                type(arguments[1]))

        if len(arguments) > 2:
            if not type_group_generator.Integer == arguments[2]:
                third_is_trunc = TypeModifiers.has_member(localization, arguments[2], '__trunc__') and \
                                 not type_group_generator.RealNumber == type(arguments[2])

                third_is_int = TypeModifiers.has_member(localization, arguments[2], '__int__') and \
                               not type_group_generator.RealNumber == type(arguments[2])

            if third_is_trunc:
                trunc_res = TypeModifiers.type_conversion_check(localization, arguments[2], 'trunc', lambda
                    call_result: type(call_result).__name__ == 'int')

                if is_error(trunc_res):
                    return trunc_res

            if third_is_int:
                trunc_res = TypeModifiers.type_conversion_check(localization, arguments[2], 'int', lambda
                    call_result: type(call_result).__name__ == 'int')

                if is_error(trunc_res):
                    return trunc_res

            if not (third_is_trunc or third_is_int) and not type_group_generator.Integer == type(arguments[2]):
                return StypyTypeError.object_must_be_type_error(localization,
                                                                "'range' third argument", "integer", type(arguments[2]))

        ret = get_builtin_python_type_instance(localization, 'xrange')
        set_contained_elements_type(ret, get_builtin_python_type_instance(localization, 'int'))

        return ret

    @staticmethod
    def int(localization, proxy_obj, arguments):
        if len(arguments) == 0:
            return get_builtin_python_type_instance(localization, 'int')

        if type_group_generator.Str == type(arguments[0]):
            try:
                if len(arguments[0]) == 0:
                    return int()
                ret = int(arguments[0])

                return int()
            except:
                return StypyTypeError(localization, "Invalid literal for int(): '{0}'".format(arguments[0]))

        first_is_trunc = False
        second_is_trunc = False

        first_is_int = False
        second_is_int = False

        if len(arguments) > 0:
            first_is_trunc = TypeModifiers.has_member(localization, arguments[0], '__trunc__') and \
                             not type_group_generator.RealNumber == type(arguments[0])

            first_is_int = TypeModifiers.has_member(localization, arguments[0], '__int__') and \
                           not type_group_generator.RealNumber == type(arguments[0])

        if len(arguments) > 1:
            second_is_trunc = TypeModifiers.has_member(localization, arguments[1], '__trunc__') and \
                              not type_group_generator.Integer == type(arguments[1])

            second_is_int = TypeModifiers.has_member(localization, arguments[1], '__int__') and \
                            not type_group_generator.Integer == type(arguments[1])

        if first_is_trunc:
            trunc_res = TypeModifiers.type_conversion_check(localization, arguments[0], 'trunc', lambda
                call_result: type(call_result).__name__ == 'int')
            if is_error(trunc_res):
                return trunc_res

        if second_is_trunc:
            if type_group_generator.RealNumber == type(arguments[1]):
                return StypyTypeError.wrong_parameter_type_error(localization, "Integer", type(arguments[1]).__name__)
            trunc_res = TypeModifiers.type_conversion_check(localization, arguments[1], 'trunc', lambda
                call_result: type(call_result).__name__ == 'int')
            if is_error(trunc_res):
                return trunc_res

        if first_is_int:
            trunc_res = TypeModifiers.type_conversion_check(localization, arguments[0], 'int', lambda
                call_result: type(call_result).__name__ == 'int')
            if is_error(trunc_res):
                return trunc_res

        if second_is_int:
            if type_group_generator.RealNumber == type(arguments[1]):
                return StypyTypeError.wrong_parameter_type_error(localization, "Integer", type(arguments[1]).__name__)
            trunc_res = TypeModifiers.type_conversion_check(localization, arguments[1], 'int', lambda
                call_result: type(call_result).__name__ == 'int')
            if is_error(trunc_res):
                return trunc_res

        return get_builtin_python_type_instance(localization, 'int')

    @staticmethod
    def getattr(localization, proxy_obj, arguments):
        if not arguments[1] == "":
            return get_member(localization, arguments[0], arguments[1])
        else:
            return type_group_generator.DynamicType
            # atts = dir_object(arguments[0])
            # ret = None
            # for att in atts:
            #     if isinstance(att, Localization):
            #         continue
            #     ret = union_type.UnionType.add(ret, get_member(localization, arguments[0], att))
            #
            # return ret

    @staticmethod
    def oct(localization, proxy_obj, arguments):
        if type_group_generator.Integer == arguments[0]:
            return get_builtin_python_type_instance(localization, 'str')
        else:
            hex_call = get_member(localization, arguments[0], '__oct__')
            if is_error(hex_call):
                return hex_call
            call_res = invoke(localization, hex_call)
            if is_error(call_res):
                return call_res
            if type(call_res) is not str:
                return StypyTypeError.wrong_return_type_error(localization, "__oct__", type(call_res), "str")

        return get_builtin_python_type_instance(localization, 'str')

    @staticmethod
    def map(localization, proxy_obj, arguments):
        ret_type = get_builtin_python_type_instance(localization, 'list')

        if type_group_generator.Number == arguments[0]:
            set_contained_elements_type(ret_type, arguments[0]())
            return ret_type

        if is_function(arguments[0]) or is_method(arguments[0]):
            func = arguments[0]
        else:
            func = get_member(localization, arguments[0], "__call__")
            if is_error(func):
                return func

        argument_types = []

        # Collect the types of the different iterables that have been passed
        for iterable in arguments[1:]:
            # Is this an str?
            if is_str(type(iterable)):
                list_types = get_builtin_python_type_instance(localization, type(iterable).__name__)
            else:
                if can_store_elements(iterable):
                    list_types = get_contained_elements_type(iterable)
                else:
                    return StypyTypeError.object_must_be_type_error(localization, "Map second parameter", "iterable")

            argument_types.append(list_types)

        inv_type = invoke(localization, func, *argument_types)

        if is_error(inv_type):
            return inv_type
        else:
            set_contained_elements_type(ret_type, inv_type)
        return ret_type

    @staticmethod
    def zip(localization, proxy_obj, arguments):
        ret_type = get_builtin_python_type_instance(localization, 'list')
        func = arguments[0]
        tuple_ = get_builtin_python_type_instance(localization, 'tuple')
        contents = None
        for iterable in arguments:
            if is_str(type(iterable)):
                list_types = get_builtin_python_type_instance(localization, type(iterable).__name__)
            else:
                if can_store_elements(iterable):
                    list_types = get_contained_elements_type(iterable)
                else:
                    return StypyTypeError.object_must_be_type_error(localization, "Zip arguments", "iterable")

            if is_undefined(list_types):
                return ret_type

            if is_error(list_types):
                return list_types

            contents = union_type.UnionType.add(contents, list_types)

        tuple_ = get_builtin_python_type_instance(localization, "tuple")
        tuple_.set_contained_type(contents)
        set_contained_elements_type(ret_type, tuple_)

        return ret_type

    @staticmethod
    def reversed(localization, proxy_obj, arguments):
        ret = reversed(arguments[0])
        wrap = StandardWrapper(ret)
        if is_str(type(arguments[0])):
            content = get_builtin_python_type_instance(localization, "str")
        else:
            content = get_contained_elements_type(arguments[0])
        set_contained_elements_type(wrap, content)

        return wrap

    @staticmethod
    def tuple(localization, proxy_obj, arguments):
        if len(arguments) == 0:
            return get_builtin_python_type_instance(localization, 'tuple')
        else:
            return get_builtin_python_type_instance(localization, 'tuple', arguments[0])

    @staticmethod
    def frozenset(localization, proxy_obj, arguments):
        ret = get_builtin_python_type_instance(localization, 'frozenset')
        if len(arguments) == 0:
            return ret

        param = arguments[0]
        if is_str(type(param)):
            set_contained_elements_type(ret,
                                        get_builtin_python_type_instance(localization, type(param).__name__))
        else:
            set_contained_elements_type(ret, get_contained_elements_type(param))

        return ret

    @staticmethod
    def sorted(localization, proxy_obj, arguments):
        ret = get_builtin_python_type_instance(localization, 'list')

        param = arguments[0]
        if is_str(type(param)):
            contained_type = get_builtin_python_type_instance(localization, 'str')
            set_contained_elements_type(ret,
                                        contained_type)
        else:
            contained_type = get_contained_elements_type(param)
            set_contained_elements_type(ret, contained_type)

        if len(arguments) >= 2:
            if type(arguments[1]) is types.FunctionType:
                call_arg1 = invoke(localization, arguments[1],
                                   contained_type,
                                   contained_type)
            else:
                call_arg1 = invoke(localization, get_member(localization, arguments[1], "__call__"),
                                   contained_type,
                                   contained_type)
            if is_error(call_arg1):
                return call_arg1

        if len(arguments) >= 3:
            if type(arguments[2]) is types.FunctionType:
                call_arg2 = invoke(localization, arguments[2],
                                   contained_type)
                if is_error(call_arg2):
                    return call_arg2
            else:
                if has_member(localization, arguments[2], "__call__"):
                    call_arg2 = invoke(localization, get_member(localization, arguments[2], "__call__"),
                                       contained_type)
                    if is_error(call_arg2):
                        return call_arg2

        return ret

    @staticmethod
    def super(localization, proxy_obj, arguments):
        ret = get_builtin_python_type_instance(localization, 'super')
        if len(arguments) == 1:
            ret = super(arguments[0])
        if len(arguments) == 2:
            if (type(arguments[0]) is types.TypeType) and (arguments[1] is types.NoneType):
                return super(arguments[0], None)
            if (type(arguments[0]) is types.TypeType) and (type(arguments[1]) is types.InstanceType):
                if type(arguments[1]) is arguments[0]:
                    ret = super(arguments[0], arguments[1])
                else:
                    ret = StypyTypeError.object_must_be_type_error(localization, "super second parameter",
                                                                   type(arguments[0]).__name__)
            else:
                if isinstance(arguments[1], StandardWrapper):
                    obj_1 = arguments[1].get_wrapped_type()
                else:
                    obj_1 = arguments[1]
                if not isinstance(obj_1, arguments[0]):
                    ret = StypyTypeError(localization,
                                         "obj must be an instance or subtype of type")
                    # # super(type, instance of type or None)
                    # if (type(arguments[1]) is types.InstanceType and
                    #         (type(arguments[1]) is arguments[0])) or type(arguments[1] is types.NoneType):
                    #     ret = super(type(arguments[0]), arguments[1])
                    # else:
                    #     if ((type(arguments[1]) is not types.TypeType) and
                    #             (type(arguments[1]) is arguments[0])):
                    #         ret = super(arguments[0], arguments[1])
                    #     else:
                    #         ret = StypyTypeError.object_must_be_type_error(localization, "super second parameter",
                    #                                                        type(arguments[0]).__name__)
                else:
                    ret = super(arguments[0], obj_1)
        return ret

    @staticmethod
    def hasattr(localization, proxy_obj, arguments):
        if arguments[1] == "":
            Advice.value_not_defined_advice(localization, "hasattr", "<member to get>")
        else:
            return has_member(localization, arguments[0], arguments[1])
        return False

    @staticmethod
    def delattr(localization, proxy_obj, arguments):
        if arguments[1] == "":
            Advice.value_not_defined_advice(localization, "delattr", "<member to delete>")
        else:
            try:
                return del_member(localization, arguments[0], arguments[1])
            except Exception as ex:
                return StypyTypeError.member_cannot_be_deleted_error(localization, arguments[0], arguments[1], str(ex))
        return types.NoneType

    @staticmethod
    def setattr(localization, proxy_obj, arguments):
        if arguments[1] == "":
            Advice.value_not_defined_advice(localization, "setattr", "<member to set>")
        else:
            try:
                return set_member(localization, arguments[0], arguments[1], arguments[2])
            except Exception as ex:
                return StypyTypeError.member_cannot_be_set_error(localization, arguments[0], arguments[1], arguments[2],
                                                                 str(ex))
        return types.NoneType

    @staticmethod
    def str(localization, proxy_obj, arguments):
        str_func = None

        if len(arguments) > 0:
            if has_member(localization, arguments[0], '__str__'):
                str_func = get_member(localization, arguments[0], '__str__')
            else:
                if has_member(localization, arguments[0], 'stypy__str__'):
                    str_func = get_member(localization, arguments[0], 'stypy__str__')

            if str_func is None:
                return StypyTypeError.member_do_not_exist_error(localization, "__str__", arguments[0])

            ret = invoke(localization, str_func)
            if not is_error(ret):
                if type(ret) is not str:
                    return StypyTypeError.wrong_return_type_error(localization, "__str__", type(ret), "string")

        return get_builtin_python_type_instance(localization, "str")

    @staticmethod
    def iter(localization, proxy_obj, arguments):
        param = arguments[0]

        # strings
        if is_str(type(param)) and len(arguments) == 1:
            str_iter = get_builtin_python_type_instance(localization, 'iterator', ' ')
            set_contained_elements_type(str_iter,
                                        get_builtin_python_type_instance(localization, type(param).__name__))
            return str_iter
        else:
            if len(arguments) == 2:
                callable_ = get_builtin_python_type_instance(localization, "callable_iterator")
                if not callable(param):
                    return StypyTypeError(localization, "iter(v, w): v must be callable")

                if type(param) is types.ClassType:
                    instance = invoke(localization, param)
                    if is_error(instance):
                        return instance
                    set_contained_elements_type(callable_, instance)

                if type(param) is types.InstanceType:
                    instance = invoke(localization, get_member(localization, param, "__call__"))
                    if is_error(instance):
                        StypyTypeError.remove_error_msg(instance)
                        set_contained_elements_type(callable_, param)
                    else:
                        set_contained_elements_type(callable_, instance)

                return callable_

            # Elements that define __iter__
            iter_method = get_member(localization, param, "__iter__")
            # contained = get_contained_elements_type(param)
            if not is_error(iter_method):
                ret = invoke(localization, iter_method)
                if not is_error(ret):
                    # set_contained_elements_type(ret, ret)
                    return ret
                else:
                    StypyTypeError.remove_error_msg(ret)

            return StypyTypeError.object_must_be_type_error(localization, "{0} object".format(param), "iterable")

    @staticmethod
    def range(localization, proxy_obj, arguments):
        first_is_trunc = False
        second_is_trunc = False
        third_is_trunc = False

        first_is_int = False
        second_is_int = False
        third_is_int = False

        if len(arguments) > 0:
            if not type_group_generator.Integer == arguments[0]:
                first_is_trunc = TypeModifiers.has_member(localization, arguments[0], '__trunc__') and \
                                 not type_group_generator.RealNumber == type(arguments[0])

                first_is_int = TypeModifiers.has_member(localization, arguments[0], '__int__') and \
                               not type_group_generator.RealNumber == type(arguments[0])

            if first_is_trunc:
                trunc_res = TypeModifiers.type_conversion_check(localization, arguments[0], 'trunc', lambda
                    call_result: type(call_result).__name__ == 'int')
                if is_error(trunc_res):
                    return trunc_res

            if first_is_int:
                trunc_res = TypeModifiers.type_conversion_check(localization, arguments[0], 'int', lambda
                    call_result: type(call_result).__name__ == 'int')
                if is_error(trunc_res):
                    return trunc_res

            if not (first_is_trunc or first_is_int) and not type_group_generator.Integer == type(arguments[0]):
                return StypyTypeError.object_must_be_type_error(localization,
                                                                "'range' first argument", "integer", type(arguments[0]))

        if len(arguments) > 1:
            if not type_group_generator.Integer == arguments[1]:
                second_is_trunc = TypeModifiers.has_member(localization, arguments[1], '__trunc__') and \
                                  not type_group_generator.RealNumber == type(arguments[1])

            if not type_group_generator.Integer == arguments[1]:
                second_is_int = TypeModifiers.has_member(localization, arguments[1], '__int__') and \
                                not type_group_generator.RealNumber == type(arguments[1])

            if second_is_trunc:
                trunc_res = TypeModifiers.type_conversion_check(localization, arguments[1], 'trunc', lambda
                    call_result: type(call_result).__name__ == 'int')
                if is_error(trunc_res):
                    return trunc_res

            if second_is_int:
                trunc_res = TypeModifiers.type_conversion_check(localization, arguments[1], 'int', lambda
                    call_result: type(call_result).__name__ == 'int')
                if is_error(trunc_res):
                    return trunc_res

            if not (second_is_trunc or second_is_int) and not type_group_generator.Integer == type(arguments[1]):
                return StypyTypeError.object_must_be_type_error(localization,
                                                                "'range' second argument", "integer",
                                                                type(arguments[1]))

        if len(arguments) > 2:
            if not type_group_generator.Integer == arguments[2]:
                third_is_trunc = TypeModifiers.has_member(localization, arguments[2], '__trunc__') and \
                                 not type_group_generator.RealNumber == type(arguments[2])

                third_is_int = TypeModifiers.has_member(localization, arguments[2], '__int__') and \
                               not type_group_generator.RealNumber == type(arguments[2])

            if third_is_trunc:
                trunc_res = TypeModifiers.type_conversion_check(localization, arguments[2], 'trunc', lambda
                    call_result: type(call_result).__name__ == 'int')

                if is_error(trunc_res):
                    return trunc_res

            if third_is_int:
                trunc_res = TypeModifiers.type_conversion_check(localization, arguments[2], 'int', lambda
                    call_result: type(call_result).__name__ == 'int')

                if is_error(trunc_res):
                    return trunc_res

            if not (third_is_trunc or third_is_int) and not type_group_generator.Integer == type(arguments[2]):
                return StypyTypeError.object_must_be_type_error(localization,
                                                                "'range' third argument", "integer", type(arguments[2]))

        ret = get_builtin_python_type_instance(localization, 'list')
        set_contained_elements_type(ret, get_builtin_python_type_instance(localization, 'int'))

        return ret

    @staticmethod
    def property(localization, proxy_obj, arguments):
        ret = None

        if len(arguments) == 0:
            return property()

        def get(obj):
            return arguments[0](obj, localization)

        def set(obj, value):
            return arguments[1](obj, localization, value)

        def del_(obj):
            return arguments[2](obj, localization)

        # Kwargs present
        if type(arguments[-1]) is dict:
            kwargs = arguments[-1]
            arguments = arguments[:-1]
            if len(arguments) == 0:
                ret = property(**kwargs)
            if len(arguments) == 1:
                ret = property(get, **kwargs)
            if len(arguments) == 2:
                ret = property(get, set, **kwargs)
            if len(arguments) == 3:
                ret = property(get, set, del_, **kwargs)
            return ret

        if len(arguments) > 0:
            ret = property(get)
        if len(arguments) > 1:
            ret = property(get, set)
        if len(arguments) > 2:
            ret = property(get, set, del_)
        if len(arguments) > 3:
            ret = property(get, set, del_, arguments[3])

        return ret

    @staticmethod
    def reload(localization, proxy_obj, arguments):
        return arguments[0]

    @staticmethod
    def dir(localization, proxy_obj, arguments):
        ret = get_builtin_python_type_instance(localization, 'list')

        contents = get_builtin_python_type_instance(localization, 'str')
        set_contained_elements_type(ret, contents)
        return ret

    @staticmethod
    def dict(localization, proxy_obj, arguments):
        ret = get_builtin_python_type_instance(localization, 'dict')
        if len(arguments) == 0:
            return ret
        if can_store_keypairs(arguments[0]):
            return arguments[0]
        else:
            contents = get_contained_elements_type(arguments[0])
            if not compare_type(contents, tuple):
                return StypyTypeError.object_must_be_type_error(localization,
                                                                'Iterable argument to build a dictionary',
                                                                '(key,value) tuple')
            else:
                keys = get_contained_elements_type(contents)
                values = keys
                if isinstance(keys, union_type.UnionType):
                    keys = keys.types
                else:
                    keys = [keys]

                for key in keys:
                    set_contained_elements_type_for_key(ret, key, values)

                return ret

    @staticmethod
    def next(localization, proxy_obj, arguments):
        next_method = get_member(localization, arguments[0], "next")
        elements = invoke(localization, next_method)

        if len(arguments) > 1:
            elements = union_type.UnionType.add(elements, arguments[1])

        return elements

    @staticmethod
    def eval(localization, proxy_obj, arguments):
        TypeWarning.enable_usage_of_dynamic_types_warning(localization, "eval")
        return type_group_generator.DynamicType()

    @staticmethod
    def execfile(localization, proxy_obj, arguments):
        TypeWarning.enable_usage_of_dynamic_types_warning(localization, "execfile")
        return type_group_generator.DynamicType()

    @staticmethod
    def complex(localization, proxy_obj, arguments):
        if len(arguments) == 0:
            return complex()
        if type_group_generator.Str == type(arguments[0]):
            try:
                if len(arguments[0]) == 0:
                    return complex()
                ret = complex(arguments[0])

                return complex()
            except:
                return StypyTypeError(localization, "complex() arg is a malformed string : '{0}'".format(arguments[0]))

        first_is_float = False
        second_is_float = False

        first_is_complex = False
        second_is_complex = False

        if len(arguments) > 0:
            first_is_float = TypeModifiers.has_member(localization, arguments[0], '__float__') and \
                             not type_group_generator.RealNumber == type(arguments[0])

            first_is_complex = TypeModifiers.has_member(localization, arguments[0], '__complex__') and \
                               not type_group_generator.RealNumber == type(arguments[0])

        if len(arguments) > 1:
            second_is_float = TypeModifiers.has_member(localization, arguments[1], '__float__') and \
                              not type_group_generator.Integer == type(arguments[1])

            second_is_complex = TypeModifiers.has_member(localization, arguments[1], '__complex__') and \
                                not type_group_generator.Integer == type(arguments[1])

        if first_is_float:
            trunc_res = TypeModifiers.type_conversion_check(localization, arguments[0], 'float', lambda
                call_result: type(call_result).__name__ == 'float')
            if is_error(trunc_res):
                return trunc_res

        if second_is_float:
            if type_group_generator.RealNumber == type(arguments[1]):
                return StypyTypeError.wrong_parameter_type_error(localization, "Integer", type(arguments[1]).__name__)
            trunc_res = TypeModifiers.type_conversion_check(localization, arguments[1], 'float', lambda
                call_result: type(call_result).__name__ == 'float')
            if is_error(trunc_res):
                return trunc_res

        if first_is_complex:
            trunc_res = TypeModifiers.type_conversion_check(localization, arguments[0], 'complex', lambda
                call_result: type(call_result).__name__ == 'float' or type(call_result).__name__ == 'complex')
            if is_error(trunc_res):
                return trunc_res

        if second_is_complex:
            if type_group_generator.RealNumber == type(arguments[1]):
                return StypyTypeError.wrong_parameter_type_error(localization, "Integer", type(arguments[1]).__name__)
            trunc_res = TypeModifiers.type_conversion_check(localization, arguments[1], 'complex', lambda
                call_result: type(call_result).__name__ == 'float' or type(call_result).__name__ == 'complex')
            if is_error(trunc_res):
                return trunc_res

        return complex()

    @staticmethod
    def ord(localization, proxy_obj, arguments):
        arg = arguments[0]
        # No value, cannot check :(
        if len(arg) == 0:
            return get_builtin_python_type_instance(localization, 'int')
        else:
            if len(arg) == 1:
                return get_builtin_python_type_instance(localization, 'int')
            else:
                return StypyTypeError(localization, "string parameter of the ord() function must be of length 1")

    @staticmethod
    def __import__(localization, proxy_obj, arguments):
        if arguments is "":
            TypeWarning.enable_usage_of_dynamic_types_warning(localization, "__import__")
            return type_group_generator.DynamicType
        name = arguments[0]
        if name is "":
            TypeWarning.enable_usage_of_dynamic_types_warning(localization, "__import__")
            return type_group_generator.DynamicType

        name_list = []
        for arg in arguments[1:]:
            if isinstance(arg, list):
                for elem in arg:
                    if elem is "":
                        TypeWarning.enable_usage_of_dynamic_types_warning(localization, "__import__")
                    else:
                        name_list.append(elem)

        dest_type_store = contexts.context.Context.get_current_active_context_for_module(localization.file_name)
        if is_python_library_module(name):
            return import_from_module(localization, name, None, dest_type_store, *name_list)
        else:
            try:
                import sys
                __import__(name)
                mod_obj = sys.modules[name]
                if hasattr(mod_obj, 'module_type_store'):
                    or_type_store = mod_obj.module_type_store
                    import_from_module(localization, name, or_type_store, dest_type_store, *name_list)
                    return dest_type_store
                if hasattr(mod_obj, '__file__'):
                    if mod_obj.__file__.endswith('.pyd'):
                        # import_pyd(localization, name, dest_type_store, wrap_contained_type, name_list)
                        return sys.modules[name]

                TypeWarning.enable_usage_of_dynamic_types_warning(localization, "__import__")
                return type_group_generator.DynamicType
            except Exception as e:
                return StypyTypeError(localization, "Cannot import module {0}: {1}".format(name, str(e)))

    @staticmethod
    def len(localization, proxy_obj, arguments):
        if isinstance(arguments[0], StandardWrapper):
            if isinstance(arguments[0].get_wrapped_type(), dict):
                return int()
            if isinstance(arguments[0].get_wrapped_type(), list):
                return int()

            # wrapped = arguments[0].get_wrapped_type()
            if has_member(localization, arguments[0].get_wrapped_type(), '__len__'):
                return int()
            else:
                return StypyTypeError(localization, "len argument have no size")

        return int()
