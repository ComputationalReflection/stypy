import time
import operator

from known_python_types_copy import known_python_type_typename_samplevalues, Foo

# Predefined instances for some types that do not have a non-parameter constructor of an alternative way to create
# an instance. This is used when needing fake values for types
__known_instances = {
    UnicodeEncodeError: UnicodeEncodeError("a", u"b", 1, 2, "e"),
    UnicodeDecodeError: UnicodeDecodeError("a", "b", 1, 2, "e"),
    UnicodeTranslateError: UnicodeTranslateError(u'0', 1, 2, '3'),
    type(time.gmtime()): time.gmtime(),
    operator.attrgetter: operator.attrgetter(Foo.qux),
    operator.methodcaller: operator.methodcaller(Foo.bar),
}


# TODO: This needs to be completed
def get_type_sample_value(type_):
    try:
        if type_ in __known_instances:
            return __known_instances[type_]

        if type_ in known_python_type_typename_samplevalues:
            return known_python_type_typename_samplevalues[type_][1]

        return type_()
    except TypeError as t:
        pass
