from ...python_types_copy.non_python_type_copy import NonPythonType


class UndefinedType(NonPythonType):
    """
    The type of an undefined variable
    """

    def __str__(self):
        return 'Undefined'

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return isinstance(other, UndefinedType)
