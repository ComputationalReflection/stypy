from type_error_copy import TypeError


class UndefinedTypeError(TypeError):
    """
    Child class of TypeError to model an special type of error: A variable has a type that cannot be determined.
    """

    def __init__(self, localization, msg, prints_msg=True):
        TypeError.__init__(self, localization, msg, prints_msg)
        #super(UndefinedTypeError, self).__init__(localization, msg, prints_msg)