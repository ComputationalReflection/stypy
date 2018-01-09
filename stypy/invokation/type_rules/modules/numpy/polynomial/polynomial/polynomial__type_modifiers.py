

import types

import numpy

from stypy.errors.type_error import StypyTypeError
from stypy.invokation.handlers import call_utilities
from stypy.invokation.type_rules.type_groups.type_group_generator import Integer, IterableDataStructure, \
    IterableDataStructureWithTypedElements, Str, Number
from stypy.type_inference_programs.stypy_interface import set_contained_elements_type, wrap_type, \
    get_contained_elements_type
from stypy.types.type_wrapper import TypeWrapper
from stypy.types.union_type import UnionType


class TypeModifiers:
    @staticmethod
    def Polynomial(localization, proxy_obj, arguments):
        if Integer == type(arguments[0]):
            return proxy_obj.coef.min()

        dvar = call_utilities.parse_varargs_and_kwargs(localization, arguments, ['coef', 'domain', 'window'], {
            'coef': IterableDataStructure,
            'domain': IterableDataStructure,
            'window': IterableDataStructure,
        }, 'Polynomial')

        if isinstance(dvar, StypyTypeError):
            return dvar

        return numpy.polynomial.polynomial.Polynomial(arguments[0].get_wrapped_type())