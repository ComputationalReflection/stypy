# !/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy

from stypy.errors.type_error import StypyTypeError
from stypy.types.known_python_types import get_sample_instance_for_type
from stypy.type_inference_programs.stypy_interface import wrap_type

from stypy.type_inference_programs.stypy_interface import get_contained_elements_type, set_contained_elements_type


class TypeModifiers:
    @staticmethod
    def iteritems(localization, proxy_obj, arguments):
        it = wrap_type(get_sample_instance_for_type("dictionary_itemiterator"))
        set_contained_elements_type(localization, it, arguments[0].get_wrapped_type().items())
        return it
