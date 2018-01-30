#!/usr/bin/env python
# -*- coding: utf-8 -*-
import collections
import types

import stypy
from stypy.errors.type_error import StypyTypeError
from stypy.errors.type_warning import TypeWarning
from stypy.reporting.localization import Localization
from stypy.types import union_type
from stypy.types.known_python_types import ExtraTypeDefinitions
from stypy.types.type_containers import get_contained_elements_type
from stypy.types.type_inspection import is_special_name_method, convert_special_name_method
from stypy.types.type_intercession import supports_intercession, has_member, get_member
from stypy.types.type_wrapper import TypeWrapper
from stypy.types.undefined_type import UndefinedType
from type_group import TypeGroup

try:
    import numpy

    short = numpy.short
    ushort = numpy.ushort
    int8 = numpy.int8
    int16 = numpy.int16
    int32 = numpy.int32
    int64 = numpy.int64

    float16 = numpy.float16
    float32 = numpy.float32
    float64 = numpy.float64
    double = numpy.double

    #complex32 = numpy.complex32
    complex64 = numpy.complex64
    complex128 = numpy.complex128
    # complex160 = numpy.complex160
    # complex256 = numpy.complex256
    # complex512 = numpy.complex512
    numpyarray = numpy.ndarray

except Exception as e:
    short = int
    ushort = int
    int8 = int
    int16 = int
    int32 = int
    int64 = int
    float16 = float
    float32 = float
    float64 = float
    float80 = float
    float96 = float
    float128 = float
    float256 = float
    double = float

    complex32 = complex
    complex64 = complex
    complex128 = complex
    complex160 = complex
    complex256 = complex
    complex512 = complex
    numpyarray = list

"""
File to define all type groups available to form type rules
"""


class DependentType(object):
    """
    A DependentType is a special base class that indicates that a type group has to be called to obtain the real
    type it represent. Call is done using the parameters that are trying to match the rule. For example, imagine that
    we call the + operator with an object that defines the __add__ method and another type to add to. With an object
    that defines an __add__ method we don't really know what will be the result of calling __add__ over this object
    with the second parameter, so the __add__ method has to be called (well, in fact, the type inference equivalent
    version of the __add__ method will be called) to obtain the real return type.

    Dependent types are a powerful mechanism to calculate the return type of operations that depend on calls to
    certain object members or even to detect incorrect definitions of members in objects (__int__ method defined in
    object that do not return int, for example).
    """

    def __init__(self, report_errors=False):
        """
        Build a Dependent type instance
        :param report_errors: Flag to indicate if errors found when calling this type will be reported or not (in that
        case other code will do it)
        """
        self.report_errors = report_errors
        self.call_arity = 0

    def __call__(self, *call_args, **call_kwargs):
        """
        Call the dependent type. Empty in this implementation, concrete calls must be defined in subclasses
        """
        pass

    def set_type(self, type_):
        self.type_ = type_


"""
Type groups with special meaning. All of them define a __eq__ method that indicates if the passed type matches with
the type group, storing this passed type. They also define a __call__ method that actually perform the type checking
and calculate the return type. __eq__ and __call__ methods are called sequentially if __eq__ result is True, so the
storage of the passed type is safe to use in the __call__ as each time an __eq__ is called is replaced. This is the
way the type rule checking mechanism works: TypeGroups are not meant to be used in other parts of the stypy runtime,
and if they do, only the __eq__ method should be used to check if a type belongs to a group.
"""


class HasMember(TypeGroup, DependentType):
    """
        Type of any object that has a member with the specified arity, and that can be called with the corresponding
        params.
    """

    def __init__(self, member, expected_return_type, call_arity, report_errors=True):
        DependentType.__init__(self, report_errors)
        TypeGroup.__init__(self, [])
        self.member = member
        self.expected_return_type = expected_return_type
        self.member_obj = None
        self.call_arity = call_arity

    def format_arity(self):
        str_ = "("
        for i in xrange(self.call_arity):
            str_ += "param" + str(i) + ", "

        if self.call_arity > 0:
            str_ = str_[:-2]

        return str_ + ")"

    def __eq__(self, type_):
        if is_special_name_method(self.member):
            converted_name = convert_special_name_method(self.member)
            if has_member(Localization.get_current(), type_, converted_name):
                self.member_obj = get_member(Localization.get_current(), type_, converted_name)
                return True
            else:
                self.member_obj = None

        if has_member(Localization.get_current(), type_, self.member):
            self.member_obj = get_member(Localization.get_current(), type_, self.member)
            return True
        else:
            self.member_obj = None

        return False

    def __call__(self, localization, *call_args, **call_kwargs):
        if callable(self.member_obj):
            # Final call arguments are composed by the self object plust the actual call arguments
            final_call_args = list(call_args)  # [self.type_] + list(call_args)
            # Call the member
            try:
                callable_ = get_member(Localization.get_current(), self.type_, self.member_obj.__name__)
            except:
                callable_ = self.type_

            # Call was impossible: The member do not exist
            if isinstance(callable_, StypyTypeError):
                StypyTypeError.remove_error_msg(callable_)
                self.member_obj = None
                return False, callable_

            equivalent_type = stypy.type_inference_programs.stypy_interface.invoke(localization, callable_,
                                                                                   *final_call_args,
                                                                                   **call_kwargs)

            # Call was impossible: Invokation error has to be removed because we provide a general one later
            if isinstance(equivalent_type, StypyTypeError):
                if not self.report_errors:
                    StypyTypeError.remove_error_msg(equivalent_type)
                self.member_obj = None
                return False, equivalent_type

            # Call was possible, but the expected return type cannot be predetermined (we have to recheck it later)
            if self.expected_return_type is UndefinedType:
                self.member_obj = None
                return True, equivalent_type

            # Call was possible, but the expected return type is Any)
            if self.expected_return_type is DynamicType:
                self.member_obj = None
                return True, equivalent_type

            # Call was possible, but the expected return type can be of multiple types)
            if type(self.expected_return_type) is list:
                for expected in self.expected_return_type:
                    # Call was possible, so we check if the predetermined return type is the same that the one that is
                    # returned
                    if issubclass(type(equivalent_type), expected):
                        self.member_obj = None
                        return True, None

                self.member_obj = None
                return False, equivalent_type

            # Call was possible, so we check if the predetermined return type is the same that the one that is returned
            if not issubclass(type(equivalent_type), self.expected_return_type):
                self.member_obj = None
                return False, equivalent_type
            else:
                return True, equivalent_type

        self.member_obj = None
        return True, None

    def __repr__(self):
        ret_str = "<Any Object>."
        ret_str += str(self.member)
        ret_str += self.format_arity()

        if self.expected_return_type is not DynamicType:
            if isinstance(self.expected_return_type, list):
                ret_str += ": " + str(self.expected_return_type)
            else:
                if hasattr(self.expected_return_type, '__name__'):
                    ret_str += ": " + self.expected_return_type.__name__

        return ret_str


class DontHaveMember(TypeGroup, DependentType):
    """
        Type of any object that dont have members with the specified arity.
    """

    def __init__(self, members, call_arity, report_errors=False):
        DependentType.__init__(self, report_errors)
        TypeGroup.__init__(self, [])
        self.members = members
        self.member_obj = None
        self.call_arity = call_arity

    def format_arity(self):
        str_ = "("
        for i in xrange(self.call_arity):
            str_ += "parameter" + str(i) + ", "

        if self.call_arity > 0:
            str_ = str_[:-2]

        return str_ + ")"

    def __eq__(self, type_):
        for member in self.members:
            if is_special_name_method(member):
                converted_name = convert_special_name_method(member)
                if has_member(Localization.get_current(), type_, converted_name):
                    return False

            if has_member(Localization.get_current(), type_, member):
                return False

        return True

    def __call__(self, localization, *call_args, **call_kwargs):
        self.member_obj = None
        return True, None

    def __repr__(self):
        ret_str = "Instance not defining the members "
        ret_str += str(self.members)

        return ret_str


class IterableDataStructureWithTypedElements(TypeGroup, DependentType):
    """
    Represent all iterable data structures that contain a certain type or types
    """

    def __init__(self, *content_types):
        DependentType.__init__(self, True)
        TypeGroup.__init__(self, [])
        self.content_types = content_types
        self.type_ = None
        self.call_arity = 0

    def __eq__(self, type_):
        self.type_ = type_
        return type_ in TypeGroups.IterableDataStructure

    @staticmethod
    def __get_type_of(obj):
        if type(obj) is types.InstanceType:
            return obj
        if obj is UndefinedType:
            return UndefinedType
        if isinstance(obj, TypeWrapper):
            return type(obj.wrapped_type)

        return type(obj)

    def __call__(self, localization, *call_args, **call_kwargs):
        contained_elements = get_contained_elements_type(self.type_)
        if isinstance(contained_elements, union_type.UnionType):
            types_to_examine = contained_elements.types
        else:
            types_to_examine = [contained_elements]

        right_types = []
        wrong_types = []

        for type_ in types_to_examine:
            match_found = False
            for declared_contained_type in self.content_types:
                if declared_contained_type == IterableDataStructureWithTypedElements.__get_type_of(type_):
                    if isinstance(declared_contained_type, DependentType):
                        declared_contained_type.set_type(type_)
                        if hasattr(declared_contained_type, 'member'):
                            declared_contained_type.member_obj = getattr(type_, declared_contained_type.member)
                        if declared_contained_type.call_arity == 0:
                            correct, return_type = declared_contained_type(localization)
                        else:
                            correct, return_type = declared_contained_type(localization, type_)
                        if correct:
                            match_found = True
                            if type_ not in right_types:
                                right_types.append(type_)
                                if type_ in wrong_types:
                                    wrong_types.remove(type_)
                        else:
                            if type_ not in wrong_types and type_ not in right_types:
                                wrong_types.append(type_)
                    else:
                        match_found = True
                        right_types.append(type_)

            if not match_found:
                if type_ not in wrong_types and type_ not in right_types:
                    wrong_types.append(type_)

        if self.report_errors:
            # All types are wrong
            if len(right_types) == 0:
                if len(wrong_types) > 0:
                    StypyTypeError(localization,
                                   "None of the iterable contained types: {0} match the expected ones {1}".format(
                                       str(types_to_examine), str(self.content_types)
                                   ))
            else:
                if len(wrong_types) > 0:
                    TypeWarning(localization,
                                "Some of the iterable contained types: {0} do not match the expected ones {1}".format(
                                    str(wrong_types), str(self.content_types)
                                ))
        else:
            if len(right_types) == 0 and len(wrong_types) > 0:
                TypeWarning(localization,
                            "Some of the iterable contained types: {0} do not match the expected ones {1}".format(
                                str(wrong_types), str(self.content_types)
                            ))

        if len(right_types) > 0:
            return True, None
        else:
            return False, wrong_types

    def __repr__(self):
        ret_str = "Iterable["

        contents = ""
        for content in self.content_types:
            contents += str(content) + ", "
        contents = contents[:-2]

        ret_str += contents
        ret_str += "]"
        return ret_str


class DynamicType(TypeGroup):
    """
    Any type (type cannot be statically calculated)
    """

    def __init__(self, *members):
        TypeGroup.__init__(self, [])
        self.members = members

    def __eq__(self, type_):
        return True


class SupportsStructuralIntercession(TypeGroup):
    """
    Any Python object that supports structural intercession
    """

    def __init__(self, *members):
        TypeGroup.__init__(self, [])
        self.members = members

    def __eq__(self, type_):
        self.type_ = type_
        return supports_intercession(type_)

    def __call__(self, localization, *call_args, **call_kwargs):
        temp = self.type_
        self.type_ = None

        return temp


class SubtypeOf(TypeGroup):
    """
    A subtype of the type passed in the constructor
    """

    def __init__(self, *types_):
        TypeGroup.__init__(self, [])
        self.types = types_

    def __eq__(self, type_):
        self.type_ = type_
        for pattern_type in self.types:
            if not issubclass(type_, pattern_type):
                return False
        return True

    def __call__(self, localization, *call_args, **call_kwargs):
        temp = self.type_
        self.type_ = None

        return temp


class IsHashable(TypeGroup):
    """
    Represent types that can properly implement the __hash__ members, so it can be placed as keys on a dict
    """

    def __init__(self, *types_):
        TypeGroup.__init__(self, [])
        self.types = types_

    def __eq__(self, type_):
        self.type_ = type_
        if issubclass(type_, collections.Hashable):
            return True
        return False

    def __call__(self, localization, *call_args, **call_kwargs):
        temp = self.type_
        self.type_ = None

        return temp


class TypeOfParam(TypeGroup, DependentType):
    """
    This type group is special in the sense that it don't really group any types, only returns the param number
    passed in the constructor when it is called with a list of parameters. This is really used to simplify several
    type rules in which the type returned by a member call is equal to the type of one of its parameters
    """

    def __init__(self, *param_number):
        DependentType.__init__(self)
        TypeGroup.__init__(self, [])
        self.param_number = param_number[0]

    def __eq__(self, type_):
        return False

    def __repr__(self):
        ret_str = type(self).__name__ + "(" + str(self.param_number) + ")"

        return ret_str

    def __call__(self, localization, *call_args, **call_kwargs):
        return call_args[0][self.param_number - 1]


class TypeObjectOfParam(TypeGroup, DependentType):
    """
    Returns the type object of the specified parameter number
    """

    def __init__(self, *param_number):
        DependentType.__init__(self)
        TypeGroup.__init__(self, [])
        self.param_number = param_number[0]

    def __eq__(self, type_):
        return False

    def __repr__(self):
        ret_str = type(self).__name__ + "(" + self.param_number + ")"

        return ret_str

    def __call__(self, localization, *call_args, **call_kwargs):
        param = call_args[0][self.param_number - 1]
        if not isinstance(param, TypeWrapper):
            ret = type(param)
        else:
            ret = type(param.wrapped_type)
        return ret


class Callable(TypeGroup):
    """
    Represent all callable objects (those that define the member __call__)
    """

    def __init__(self):
        TypeGroup.__init__(self, [])

    def __eq__(self, type_):
        self.member_obj = hasattr(type_, "__call__")
        return self.member_obj

    def __call__(self, localization, *call_args, **call_kwargs):
        temp = self.member_obj
        self.member_obj = None

        return temp


class TypeObject(TypeGroup):
    """
    Represent type and types.ClassType types
    """
    type_objs = [type, types.ClassType]

    def __init__(self):
        TypeGroup.__init__(self, [])

    def __eq__(self, type_):
        self.member_obj = type_
        return self.member_obj in TypeObject.type_objs

    def __call__(self, localization, *call_args, **call_kwargs):
        temp = self.member_obj
        self.member_obj = None

        return temp


class InstanceOfType(TypeGroup):
    """
    Represent type and types.ClassType types
    """
    type_objs = [type, types.ClassType]

    def __init__(self):
        TypeGroup.__init__(self, [])

    def __eq__(self, type_):
        self.member_obj = type(type_)
        return self.member_obj in TypeObject.type_objs

    def __call__(self, localization, *call_args, **call_kwargs):
        temp = self.member_obj
        self.member_obj = None

        return temp


class VarArgType(TypeGroup):
    """
    Special type group indicating that a callable has an unlimited amount of parameters
    """

    def __init__(self, *types_):
        TypeGroup.__init__(self, [])
        self.types = types_

    def __eq__(self, type_):
        return True

    def __call__(self, localization, *call_args, **call_kwargs):
        temp = self.type_
        self.type_ = None

        return temp


class TypeGroups(object):
    """
    Class to hold definitions of type groups that are composed by lists of known Python types
    """

    def __init__(self):
        pass

    @staticmethod
    def get_rule_groups():
        """
        Obtain all the types defined in this class
        """

        def filter_func(element):
            return isinstance(element, list)

        return filter(lambda member: filter_func(getattr(TypeGroups, member)), TypeGroups.__dict__)

    # Reals
    RealNumber = [int, long, float, bool, float16, float32, float64, double]
    NumpyRealNumber = [float16, float32, float64, double]

    # Any number
    Number = [int, long, short, ushort, int8, int16, int32, int64,
              float, float16, float32, float64, double,
              bool,
              complex, complex64, complex128]#complex32, complex160, complex256, complex512]
    NumpyNumber = [short, ushort, int8, int16, int32, int64,
              float16, float32, float64, double,
              complex64, complex128]

    # Integers
    Integer = [int, long, bool, short, ushort, int8, int16, int32, int64]
    NumpyInteger = [short, ushort, int8, int16, int32, int64]

    # strings
    Str = [str, unicode, buffer]

    # Bynary strings
    ByteSequence = [buffer, bytearray, str, memoryview]

    # Data structures that can be iterable plus iterators
    IterableDataStructure = [
        list,
        dict,
        ExtraTypeDefinitions.tupleiterator,
        ExtraTypeDefinitions.dict_values,
        frozenset,
        ExtraTypeDefinitions.rangeiterator,
        types.GeneratorType,
        enumerate,
        bytearray,
        iter,
        reversed,
        ExtraTypeDefinitions.dictionary_keyiterator,
        ExtraTypeDefinitions.bytearray_iterator,
        ExtraTypeDefinitions.dictionary_valueiterator,
        ExtraTypeDefinitions.dictionary_itemiterator,
        ExtraTypeDefinitions.listiterator,
        ExtraTypeDefinitions.listreverseiterator,
        tuple,
        set,
        xrange,
        numpyarray,
    ]

    # Data structures that can be iterable plus iterators plus iterable objects that are not necessarily data structures
    IterableObject = [
        list,
        dict,
        ExtraTypeDefinitions.tupleiterator,
        ExtraTypeDefinitions.dict_values,
        frozenset,
        ExtraTypeDefinitions.rangeiterator,
        types.GeneratorType,
        enumerate,
        bytearray,
        iter,
        reversed,
        ExtraTypeDefinitions.dictionary_keyiterator,
        ExtraTypeDefinitions.bytearray_iterator,
        ExtraTypeDefinitions.dictionary_valueiterator,
        ExtraTypeDefinitions.dictionary_itemiterator,
        ExtraTypeDefinitions.listiterator,
        ExtraTypeDefinitions.listreverseiterator,
        tuple,
        set,
        xrange,
        memoryview,
        types.DictProxyType,
        numpyarray,
        file,
        ]


"""
Instances of type groups. These are the ones that are really used in the type rules, as are concrete usages
of the previously defined type groups.

NOTE: To interpret instances of type groups, you should take into account the following:

- UndefinedType as expected return type: We cannot statically determine the return
type of this method. So we obtain it calling the member, obtaining its type
and reevaluating the member ruleset again with this type substituting the dependent
one.

- DynamicType as expected return type: We also cannot statically determine the return
type of this method. But this time we directly return the return type of the invoked
member.
"""

# Type conversion methods
CastsToInt = HasMember("__int__", int, 0)
CastsToLong = HasMember("__long__", [int, long], 0)
CastsToFloat = HasMember("__float__", float, 0)
CastsToComplex = HasMember("__complex__", [float, complex], 0)
CastsToOct = HasMember("__oct__", str, 0)
CastsToHex = HasMember("__hex__", str, 0)
CastsToIndex = HasMember("__index__", int, 0)
CastsToTrunc = HasMember("__trunc__", int, 0)
CastsToCoerce = HasMember("__coerce__", tuple, 1)

# Comparison magic methods:
Overloads__cmp__ = HasMember("__cmp__", DynamicType, 1)
Overloads__eq__ = HasMember("__eq__", DynamicType, 1)
Overloads__ne__ = HasMember("__ne__", DynamicType, 1)
Overloads__lt__ = HasMember("__lt__", DynamicType, 1)
Overloads__gt__ = HasMember("__gt__", DynamicType, 1)
Overloads__le__ = HasMember("__le__", DynamicType, 1)
Overloads__ge__ = HasMember("__ge__", DynamicType, 1)

# Unary operators and functions:
Overloads__pos__ = HasMember("__pos__", DynamicType, 0)
Overloads__neg__ = HasMember("__neg__", DynamicType, 0)
Overloads__abs__ = HasMember("__abs__", DynamicType, 0)
Overloads__invert__ = HasMember("__invert__", DynamicType, 0)

Overloads__round__ = HasMember("__round__", float, 1)
Overloads__floor__ = HasMember("__floor__", float, 0)
Overloads__ceil__ = HasMember("__ceil__", float, 0)

Overloads__trunc__ = HasMember("__trunc__", int, 0)

# Normal numeric operators:
Overloads__add__ = HasMember("__add__", DynamicType, 1)
Overloads__sub__ = HasMember("__sub__", DynamicType, 1)
Overloads__mul__ = HasMember("__mul__", DynamicType, 1)
Overloads__floordiv__ = HasMember("__floordiv__", DynamicType, 1)
Overloads__div__ = HasMember("__div__", DynamicType, 1)
Overloads__truediv__ = HasMember("__truediv__", DynamicType, 1)
Overloads__mod__ = HasMember("__mod__", DynamicType, 1)
Overloads__divmod__ = HasMember("__divmod__", DynamicType, 1)
Overloads__pow__ = HasMember("__pow__", DynamicType, 2)
Overloads__lshift__ = HasMember("__lshift__", DynamicType, 1)
Overloads__rshift__ = HasMember("__rshift__", DynamicType, 1)
Overloads__and__ = HasMember("__and__", DynamicType, 1)
Overloads__or__ = HasMember("__or__", DynamicType, 1)
Overloads__xor__ = HasMember("__xor__", DynamicType, 1)

# Normal reflected numeric operators:
Overloads__radd__ = HasMember("__radd__", DynamicType, 1)
Overloads__rsub__ = HasMember("__rsub__", DynamicType, 1)
Overloads__rmul__ = HasMember("__rmul__", DynamicType, 1)
Overloads__rfloordiv__ = HasMember("__rfloordiv__", DynamicType, 1)
Overloads__rdiv__ = HasMember("__rdiv__", DynamicType, 1)
Overloads__rtruediv__ = HasMember("__rtruediv__", DynamicType, 1)
Overloads__rmod__ = HasMember("__rmod__", DynamicType, 1)
Overloads__rdivmod__ = HasMember("__rdivmod__", DynamicType, 1)
Overloads__rpow__ = HasMember("__rpow__", DynamicType, 1)
Overloads__rlshift__ = HasMember("__rlshift__", DynamicType, 1)
Overloads__rrshift__ = HasMember("__rrshift__", DynamicType, 1)
Overloads__rand__ = HasMember("__rand__", DynamicType, 1)
Overloads__ror__ = HasMember("__ror__", DynamicType, 1)
Overloads__rxor__ = HasMember("__rxor__", DynamicType, 1)

# Augmented assignment numeric operators:
Overloads__iadd__ = HasMember("__iadd__", DynamicType, 1)
Overloads__isub__ = HasMember("__isub__", DynamicType, 1)
Overloads__imul__ = HasMember("__imul__", DynamicType, 1)
Overloads__ifloordiv__ = HasMember("__ifloordiv__", DynamicType, 1)
Overloads__idiv__ = HasMember("__idiv__", DynamicType, 1)
Overloads__itruediv__ = HasMember("__itruediv__", DynamicType, 1)
Overloads__imod__ = HasMember("__imod__", DynamicType, 1)
Overloads__idivmod__ = HasMember("__idivmod__", DynamicType, 1)
Overloads__ipow__ = HasMember("__ipow__", DynamicType, 1)
Overloads__ilshift__ = HasMember("__ilshift__", DynamicType, 1)
Overloads__irshift__ = HasMember("__irshift__", DynamicType, 1)
Overloads__iand__ = HasMember("__iand__", DynamicType, 1)
Overloads__ior__ = HasMember("__ior__", DynamicType, 1)
Overloads__ixor__ = HasMember("__ixor__", DynamicType, 1)

# Class representation methods
Has__str__ = HasMember("__str__", str, 0)
Has__repr__ = HasMember("__repr__", str, 0)
Has__unicode__ = HasMember("__unicode__", unicode, 0)
Has__format__ = HasMember("__format__", str, 1)
Has__hash__ = HasMember("__hash__", int, 0)
Has__nonzero__ = HasMember("__nonzero__", bool, 0)
Has__dir__ = HasMember("__dir__", DynamicType, 0)
Has__sizeof__ = HasMember("__sizeof__", int, 0)
Has__call__ = Callable()  # HasMember("__call__", DynamicType, 0)
Has__mro__ = HasMember("__mro__", DynamicType, 0)
Has__class__ = HasMember("__class__", DynamicType, 0)
Has__dict__ = HasMember("__dict__", dict, 0)

# Collections

Has__len__ = HasMember("__len__", int, 0)
Has__getitem__ = HasMember("__getitem__", DynamicType, 1)
Has__setitem__ = HasMember("__setitem__", types.NoneType, 2)
Has__delitem__ = HasMember("__delitem__", types.NoneType, 0)
Has__iter__ = HasMember("__iter__", DynamicType, 0)
Has__reversed__ = HasMember("__reversed__", DynamicType, 0)
Has__contains__ = HasMember("__contains__", bool, 1)
Has__missing__ = HasMember("__missing__", DynamicType, 0)
Has__getslice__ = HasMember("__getslice__", DynamicType, 2)
Has__setslice__ = HasMember("__setslice__", types.NoneType, 3)
Has__delslice__ = HasMember("__delslice__", types.NoneType, 2)
Has__next = HasMember("next", DynamicType, 0)

# Context managers
Has__enter__ = HasMember("__enter__", int, 0)
Has__exit__ = HasMember("__exit__", int, 3)

# Descriptor managers
Has__get__ = HasMember("__get__", DynamicType, 1)
Has__set__ = HasMember("__set__", types.NoneType, 2)
Has__del__ = HasMember("__del__", types.NoneType, 1)

# Copying
Has__copy__ = HasMember("__copy__", DynamicType, 0)
Has__deepcopy__ = HasMember("__deepcopy__", DynamicType, 1)

# Pickling
Has__getinitargs__ = HasMember("__getinitargs__", DynamicType, 0)
Has__getnewargs__ = HasMember("__getnewargs__", DynamicType, 0)
Has__getstate__ = HasMember("__getstate__", DynamicType, 0)
Has__setstate__ = HasMember("__setstate__", types.NoneType, 1)
Has__reduce__ = HasMember("__reduce__", DynamicType, 0)
Has__reduce_ex__ = HasMember("__reduce_ex__", DynamicType, 0)

# DynamicType instance

AnyType = DynamicType()
StructuralIntercessionType = SupportsStructuralIntercession()

# Other conditions
Hashable = IsHashable()
Type = TypeObject()
TypeInstance = InstanceOfType()
VarArgs = VarArgType()
