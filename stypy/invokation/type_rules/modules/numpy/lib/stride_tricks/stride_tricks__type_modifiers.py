#!/usr/bin/env python
# -*- coding: utf-8 -*-

from stypy.errors.type_error import StypyTypeError
from stypy.invokation.handlers import call_utilities
from stypy.invokation.type_rules.type_groups.type_group_generator import Integer

from stypy.invokation.type_rules.type_groups.type_group_generator import Integer, IterableDataStructure, \
    IterableDataStructureWithTypedElements, Str, Number
from stypy.types.type_wrapper import TypeWrapper
from stypy.types.union_type import UnionType

class TypeModifiers:
    @staticmethod
    def as_strided(localization, proxy_obj, arguments):
        dvar = call_utilities.parse_varargs_and_kwargs(localization, arguments, ['shape', 'strides', 'subok', 'writeable'], {
            'shape': IterableDataStructureWithTypedElements(Integer),
            'strides': IterableDataStructureWithTypedElements(Integer),
            'subok': bool,
            'writeable': bool,
        }, 'as_strided')

        if isinstance(dvar, StypyTypeError):
            return dvar

        if 'shape' in dvar.keys():
            shape = dvar['shape'].get_wrapped_type()
        else:
            shape = None

        if 'strides' in dvar.keys():
            strides = dvar['strides'].get_wrapped_type()
        else:
            strides = None
        import numpy.lib.stride_tricks as st
        return st.as_strided(arguments[0], shape, strides)

        # t = call_utilities.get_inner_type(localization, arguments[0])
        # dims = call_utilities.get_dimensions(localization, arguments[0])
        #
        # if 'shape' in dvar.keys():
        #     shape_levels = 2
        # if shape_levels == 0:
        #     return StypyTypeError(localization,
        #                           "Invalid 'shape' parameter for reshape call: There must be at least one shape element")
        # if call_utilities.is_iterable(arguments[0]):
        #     shape_levels = 2
        #
        # if len(arguments)> 1 and not Str == type(arguments[-1]) and not Integer == type(arguments[-1]):
        #     return StypyTypeError(localization, "Invalid 'order' parameter for reshape call: {0}".format(str(arguments[-1])))
        #
        # r = TypeWrapper.get_wrapper_of(proxy_obj.__self__)
        # contained = call_utilities.get_inner_type(localization, r)
        #
        # for i in range(shape_levels):
        #     contained = call_utilities.create_numpy_array(contained)
        # return contained
        #
        # if dims > 1:
        #     return call_utilities.create_numpy_array_n_dimensions(t, dims - 1)
        #
        # return call_utilities.create_numpy_array(t)