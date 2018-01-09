import types

import numpy

from stypy.errors.type_error import StypyTypeError, Localization
from stypy.invokation.type_rules.type_groups.type_group_generator import Integer, Number, IterableDataStructure, DynamicType, RealNumber, Str
from stypy.type_inference_programs.stypy_interface import wrap_contained_type, get_contained_elements_type
from stypy.types.standard_wrapper import StandardWrapper
from stypy.types.union_type import UnionType

"""
Various functions to better process the type calculations of the calls to certain functions or methods. This is mostly
used when dealing with numpy functions.
"""


def parse_varargs(arguments, arg_names):
    """
    When a function has varargs, it takes the arguments and puts them into a dictionary whose key is the argument
    name passed in the second parameter. If a dict is at the end of the arguments array, it is automatically discarded
    as we suppose that it is the kwargs dict.
    :param arguments:
    :param arg_names:
    :return:
    """
    # Exclude kwargs
    if isinstance(arguments[-1], dict):
        arguments = arguments[:-1]

    ret = dict()
    cont = 0
    for arg in arguments:
        ret[arg_names[cont]] = arg
        cont += 1

    return ret


def __get_type(type_):
    if isinstance(type_, StandardWrapper):
        return type(type_.wrapped_type)
    if isinstance(type_, numpy.dtype):
        return type
    return type(type_)


def parse_kwargs(localization, dictionary, names_and_types, func_name, admit_Nones=True):
    """
    Takes every kwarg in the dictionary and tests if its time complies with the one specified in the third parameter.
    Contents in the dictionary and in the names_and_types array are related by name. Localization and func_name are
    provided for error messages
    :param localization:
    :param dictionary:
    :param names_and_types:
    :param func_name:
    :return:
    """
    for key in dictionary:
        if key in names_and_types.keys():
            correct_type = False
            if not isinstance(names_and_types[key], list):
                accepted_types = [names_and_types[key]]
            else:
                accepted_types = names_and_types[key]

            for type_ in accepted_types:
                if isinstance(type_, DynamicType) or type_ == __get_type(dictionary[key]) or (admit_Nones and (dictionary[key] is types.NoneType or
                                                                                     dictionary[key] is None)):
                    correct_type = True
                    break

            if not correct_type:
                return StypyTypeError(localization,
                                      "Invalid keyword parameter '{0}: {1}' passed to function {2}".format(
                                          key, type(dictionary[key]), func_name))

    return dictionary


def mergue_dicts(localization, from_dict, to_dict, func_name):
    """
    Auxiliar function to put from_dict contents into to_dict. If there are matching keys, a type error is thrown, as
    in Python we cannot pass multiple values for the same parameter.
    :param localization:
    :param from_dict:
    :param to_dict:
    :param func_name:
    :return:
    """
    for key_from in from_dict:
        if key_from in to_dict.keys():
            return StypyTypeError(localization,
                                  "'{0} got multiple values for keyword '{1}'".format(
                                      func_name, key_from))
        to_dict[key_from] = from_dict[key_from]

    return to_dict


def parse_varargs_and_kwargs(localization, arguments, arg_names, arg_names_and_types, func_name, varargpos=1):
    """
    Process function arguments supposing that we have varargs starting in the position varargpos and also kwargs.
    Test that there are no multiple values for the same argument, that the arguments match any of their valid types
    and return an error otherwise
    :return:
    """
    if len(arguments) > varargpos:
        dvar = parse_varargs(arguments[varargpos:], arg_names)
    else:
        dvar = dict()

    if isinstance(arguments[-1], dict):
        dvar = mergue_dicts(localization, dvar, arguments[-1], func_name)

    if isinstance(dvar, StypyTypeError):
        return dvar

    parse_args = parse_kwargs(localization, dvar, arg_names_and_types, func_name)

    return parse_args


def __cast_to_numpy_type(type_):
    """
    Converts a type in its equivalent numpy type (i. e.: int -> int32
    :param type_:
    :return:
    """
    try:
        import numpy

        if Integer == type(type_):
            return numpy.int32(0)
        else:
            if RealNumber == type(type_):
                return numpy.float64(0.0)
            else:
                if Number == type(type_):
                    return numpy.complex128()

    except Exception as ex:
        pass

    if type(type_) is types.EllipsisType:
        return None

    return type_


def cast_to_greater_numpy_type(type_1, type_2):
    """
    Converts two types in their equivalent numpy type, and then return the greater one.
    :param type_1:
    :param type_2:
    :return:
    """
    try:
        _t1 = cast_to_numpy_type(type_1)
        _t2 = cast_to_numpy_type(type_2)
        if _t1 is None and _t2 is None:
            return None
        if _t1 is None:
            return _t2
        if _t2 is None:
            return _t1

        # Integers vs floats
        if Integer == type(_t2) and Number == type(_t1):
            return _t1
        if Integer == type(_t1) and Number == type(_t2):
            return _t2

        #Floats vs complex
        if RealNumber == type(_t2) and Number == type(_t1):
            return _t1
        if RealNumber == type(_t1) and Number == type(_t2):
            return _t2

        if Str == type(_t2) and Number == type(_t1):
            return _t2
        if Str == type(_t1) and Number == type(_t2):
            return _t1

        if bool == type(_t2) and Number == type(_t1):
            return _t1
        if bool == type(_t1) and Number == type(_t2):
            return _t2
    except Exception as ex:
        pass
    return type_2


def cast_to_numpy_type(type_):
    """
    Turns a Python numeric type into an equivalen numpy numeric type. This is needed when turning lists or similar
    to numpy ndarrays
    :param type_:
    :return:
    """
    # Simple conversion of a single type
    if not isinstance(type_, UnionType):
        return __cast_to_numpy_type(type_)
    else:
        # Conversion of a union type
        types_in_union = type_.types
        final_type = None
        for type_u in types_in_union:
            final_type = UnionType.add(final_type, __cast_to_numpy_type(type_u))

        return final_type


def create_numpy_array(contained_type, use_list=True, dtype=None):
    """
    Creates a numpy array for the contained_type, converting it to
    :param contained_type:
    :param use_list:
    :return:
    """
    try:
        import numpy
        # If dtype has this value, we cannot use it
        if isinstance(dtype, numpy.dtype) or dtype == numpy.dtype:
            dtype = None
        if use_list:
            if dtype is not None:
                wrap = wrap_contained_type(numpy.ndarray([1], dtype=dtype))
            else:
                wrap = wrap_contained_type(numpy.ndarray([1]))
        else:
            if dtype is not None:
                wrap = wrap_contained_type(numpy.ndarray(1, dtype=dtype))
            else:
                wrap = wrap_contained_type(numpy.ndarray(1))

        if contained_type is not None:
            wrap.set_contained_type(cast_to_numpy_type(contained_type))

        wrap.overriden_members.append('shape')
        shape_tuple = wrap_contained_type(tuple([1]))
        shape_tuple.set_contained_type(int())
        wrap.shape = shape_tuple

        wrap.overriden_members.append('T')
        wrap.T = wrap

        return wrap
    except Exception as ex:
        return StypyTypeError(Localization.get_current(),
                              "Cannot create numpy array of type {0}".format(str(type(contained_type))))


def create_numpy_array_nowrappers(contained_type, use_list=True, dtype=None):
    """
    Creates a numpy array for the contained_type, converting it to
    :param contained_type:
    :param use_list:
    :return:
    """
    try:
        import numpy
        # If dtype has this value, we cannot use it
        if isinstance(dtype, numpy.dtype) or dtype == numpy.dtype:
            dtype = None

        nt = cast_to_numpy_type(contained_type)
        if use_list:
            if dtype is not None:
                wrap = numpy.ndarray([nt], dtype=dtype)
            else:
                wrap = numpy.ndarray([nt])
        else:
            if dtype is not None:
                wrap = numpy.ndarray(nt, dtype=dtype)
            else:
                wrap = numpy.ndarray(nt)

        # wrap.overriden_members.append('shape')
        # shape_tuple = wrap_contained_type(tuple([1]))
        # shape_tuple.set_contained_type(int())
        # wrap.shape = shape_tuple
        #
        # wrap.overriden_members.append('T')
        # wrap.T = wrap

        return wrap
    except Exception as ex:
        return StypyTypeError(Localization.get_current(),
                              "Cannot create numpy array of type {0}".format(str(type(contained_type))))

def check_possible_values(dvar, key, values):
    """
    Checks if the key parameter contained in the dict dvar has a value and this value is among the accepted ones.
    :param dvar:
    :param key:
    :param values:
    :return:
    """
    if isinstance(dvar, StypyTypeError):
        return dvar

    if key in dvar.keys():
        value = dvar[key]
        if value is None:
            return dvar
        if value == str():
            return dvar
        else:
            if value not in values:
                return StypyTypeError(Localization.get_current(), "Invalid value passed to parameter '{0}' ('{1}'). "
                                                                  "Only {2} are accepted".format(key, value,
                                                                                                 str(values)))

    return dvar


def check_possible_types(param, possible_types):
    """
    Checks if the type of the parameter has one of the possible types passed
    :param type_:
    :param possible_types:
    :return:
    """

    for t in possible_types:
        if t == type(param):
            return True

    return False


def check_parameter_type(dictionary, param_name, param_type):
    """
    Test if param_name exist inside dictionary and if its type is compatible with param_type
    :param dictionary:
    :param param_name:
    :param param_type:
    :return:
    """
    if param_name not in dictionary:
        return False

    return param_type == dictionary[param_name]


def is_numpy_array(obj):
    """
    Determines if an object is a numpy ndarray
    :param obj:
    :return:
    """
    if isinstance(obj, StandardWrapper):
        return isinstance(obj.wrapped_type, numpy.ndarray)
    return False


def is_python_iterable(obj):
    """
    Determines if an object is an iterable data structure from python (not including numpy)
    :param obj:
    :return:
    """
    if isinstance(obj, StandardWrapper):
        return IterableDataStructure == type(obj.wrapped_type)
    return False


def is_ndenumerate(obj):
    return type(obj).__name__ == 'ndenumerate' and 'numpy.lib.index_tricks' in type(obj).__module__


def is_ndindex(obj):
    return type(obj).__name__ == 'ndindex' and 'numpy.lib.index_tricks' in type(obj).__module__

def is_iterable(obj):
    """
    Determines if an object is an iterable data structure (including numpy)
    :param obj:
    :return:
    """
    return is_numpy_array(obj) or is_python_iterable(obj) or is_ndenumerate(obj)


def get_inner_type(localization, obj):
    ret = get_contained_elements_type(localization, obj)
    while is_iterable(ret):
        ret = get_contained_elements_type(localization, ret)

    return ret


def get_dimensions(localization, obj):
    dim = 1
    ret = get_contained_elements_type(localization, obj)
    while is_iterable(ret):
        ret = get_contained_elements_type(localization, ret)
        dim+=1

    return dim


def create_numpy_array_n_dimensions(contained_type, dims, dtype=None, use_wrappers=True):
    typ = contained_type
    for i in range(dims):
        if use_wrappers:
            typ = create_numpy_array(typ, dtype=dtype)
        else:
            typ = create_numpy_array_nowrappers(typ, dtype=dtype)
    return typ
