#!/usr/bin/env python
# -*- coding: utf-8 -*-

from stypy.invokation.type_rules.type_groups.type_group_generator import *

NumpyFloat = numpy.float64()


def same_rules_as(key):
    return type_rules_of_members[key]


type_rules_of_members = {
    'fft': [
        ((IterableDataStructure,), DynamicType),
        ((IterableDataStructure, AnyType), DynamicType),
        ((IterableDataStructure, AnyType, AnyType), DynamicType),
        ((IterableDataStructure, AnyType, AnyType, AnyType), DynamicType),
    ],
    'fftn': [
        ((Number,), DynamicType),
        ((IterableDataStructure,), DynamicType),
        ((IterableDataStructure, AnyType), DynamicType),
        ((IterableDataStructure, AnyType, AnyType), DynamicType),
        ((IterableDataStructure, AnyType, AnyType, AnyType), DynamicType),
    ],

    'ifft': lambda: same_rules_as('fft'),
    'rfft': lambda: same_rules_as('fft'),
    'irfft': lambda: same_rules_as('fft'),

    'fft2': lambda: same_rules_as('fftn'),
    'ifft2': lambda: same_rules_as('fftn'),
    'ifftn': lambda: same_rules_as('fftn'),
    'rfft2': lambda: same_rules_as('fftn'),
    'irfft2': lambda: same_rules_as('fftn'),
    'rfftn': lambda: same_rules_as('fftn'),
    'irfftn': lambda: same_rules_as('fftn'),

    'hfft': lambda: same_rules_as('fft'),

    'fftfreq': [
        ((Integer,), DynamicType),
        ((Integer, Number), DynamicType),
    ],

    'rfftfreq': lambda: same_rules_as('fftfreq'),

    'fftshift': [
        ((IterableDataStructureWithTypedElements(Number),), DynamicType),
        ((IterableDataStructureWithTypedElements(Number), Integer), DynamicType),
        ((IterableDataStructureWithTypedElements(Number), IterableDataStructureWithTypedElements(Integer)), DynamicType),
    ],

    'rfftshift': lambda: same_rules_as('fftshift'),
}