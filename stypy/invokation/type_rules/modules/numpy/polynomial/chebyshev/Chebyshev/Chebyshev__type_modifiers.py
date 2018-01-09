

import types

import numpy

from stypy.errors.type_error import StypyTypeError
from stypy.invokation.handlers import call_utilities
from stypy.invokation.type_rules.type_groups.type_group_generator import Integer, IterableDataStructure, \
    IterableDataStructureWithTypedElements, Str, Number, RealNumber
from stypy.type_inference_programs.stypy_interface import set_contained_elements_type, wrap_type, \
    get_contained_elements_type, wrap_contained_type
from stypy.types.type_wrapper import TypeWrapper
from stypy.types.union_type import UnionType


class TypeModifiers:
    @staticmethod
    def fit(localization, proxy_obj, arguments):
        dvar = call_utilities.parse_varargs_and_kwargs(localization, arguments, ['domain', 'rcond', 'full', 'w', 'window'], {
            'domain': IterableDataStructure,
            'rcond': RealNumber,
            'full': bool,
            'w': IterableDataStructure,
            'window': IterableDataStructure,
        }, 'fit', 3)

        if isinstance(dvar, StypyTypeError):
            return dvar

        ret = call_utilities.create_numpy_array_n_dimensions(call_utilities.get_inner_type(localization, arguments[0]),
                                                              call_utilities.get_dimensions(localization, arguments[0]))
        ret = numpy.polynomial.Chebyshev(ret.get_wrapped_type())

        if 'full' in dvar.keys():
            tup = wrap_contained_type(tuple())
            ld = wrap_contained_type(list())
            ld.set_contained_type(call_utilities.get_inner_type(localization, arguments[0]))
            un = UnionType.add(ret, ld)
            tup.set_contained_type(un)
            return tup

        return ret

        # if call_utilities.is_iterable(arguments[2]):
        #     return proxy_obj.im_self.fit(arguments[0].get_wrapped_type(), arguments[1].get_wrapped_type(), arguments[2].get_wrapped_type())
        #
        # return proxy_obj.im_self.fit(arguments[0].get_wrapped_type(), arguments[1].get_wrapped_type(), arguments[2])