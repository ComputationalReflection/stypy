
from stypy.invokation.type_rules.type_groups.type_group_generator import Integer
from stypy.errors.type_error import StypyTypeError
from stypy.invokation.handlers import call_utilities
from stypy.invokation.type_rules.type_groups.type_group_generator import IterableDataStructure, RealNumber
from stypy.type_inference_programs.stypy_interface import get_contained_elements_type, wrap_type, set_contained_elements_type
from stypy.types.union_type import UnionType
import numpy

class TypeModifiers:
    @staticmethod
    def poly1d(localization, proxy_obj, arguments):
        if Integer == type(arguments[0]):
            return proxy_obj(arguments[0])

        import numpy
        return call_utilities.wrap_contained_type(numpy.poly1d(arguments[0].get_wrapped_type()))


    @staticmethod
    def polyfit(localization, proxy_obj, arguments):
        dvar = call_utilities.parse_varargs_and_kwargs(localization, arguments, ['rcond', 'full', 'w', 'cov'], {
            'rcond': RealNumber,
            'full': bool,
            'w': IterableDataStructure,
            'cov': bool,
        }, 'polyfit', 3)

        if isinstance(dvar, StypyTypeError):
            return dvar

        import numpy
        return call_utilities.wrap_contained_type(numpy.polyfit(arguments[0].get_wrapped_type(), arguments[1].get_wrapped_type(), arguments[2]))