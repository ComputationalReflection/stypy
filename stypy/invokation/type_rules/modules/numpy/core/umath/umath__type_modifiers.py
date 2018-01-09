import types

from stypy.type_inference_programs.stypy_interface import get_builtin_python_type_instance
from stypy.types import union_type
from stypy.types.type_containers import set_contained_elements_type


class TypeModifiers:
    @staticmethod
    def geterrobj(localization, proxy_obj, arguments):
        ret_type = get_builtin_python_type_instance(localization, 'list')
        set_contained_elements_type(ret_type,
                                    union_type.UnionType.add(get_builtin_python_type_instance(localization, 'int'),
                                                             types.NoneType))

        return ret_type
