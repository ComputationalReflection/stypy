
from stypy.code_generation.type_inference_programs.checking.type_data_file_writer import TypeDataFileWriter

type_test = TypeDataFileWriter(__file__)

class UndefinedType:
    '\n    The type of an undefined variable\n    '

    def __str__(self):
        return ('Undefined', type_test.add_type_dict_for_context(locals()))[0]
        type_test.add_type_dict_for_context(locals())


    def __repr__(self):
        return (self.__str__(), type_test.add_type_dict_for_context(locals()))[0]
        type_test.add_type_dict_for_context(locals())


    def __eq__(self, other):
        return (isinstance(other, UndefinedType), type_test.add_type_dict_for_context(locals()))[0]
        type_test.add_type_dict_for_context(locals())


class DynamicType:
    '\n    Any type (type cannot be statically calculated)\n    '

    def __init__(self, *members):
        TypeGroup.__init__(self, [])
        self.members = members
        type_test.add_type_dict_for_context(locals())


    def __eq__(self, type_):
        return (True, type_test.add_type_dict_for_context(locals()))[0]
        type_test.add_type_dict_for_context(locals())


class BaseTypeGroup(object, ):
    '\n    All type groups inherit from this class\n    '

    def __str__(self):
        return (self.__repr__(), type_test.add_type_dict_for_context(locals()))[0]
        type_test.add_type_dict_for_context(locals())


class TypeGroup(BaseTypeGroup, ):
    '\n    A TypeGroup is an entity used in the rule files to group several Python types over a named identity. Type groups\n    are collections of types that have something in common, and Python functions and methods usually admits any of them\n    as a parameter when one of them is valid. For example, if a Python library function works with an int as the first\n    parameter, we can also use bool and long as the first parameter without runtime errors. This is for exameple a\n    TypeGroup that will be called Integer\n\n    Not all type groups are defined by collections of Python concrete types. Other groups identify Python objects with\n    a common member or structure (Iterable, Overloads__str__ identify any Python object that is iterable and any Python\n    object that has defined the __str__ method properly) or even class relationships (SubtypeOf type group only matches\n    with classes that are a subtype of the one specified.\n\n    Type groups are the workhorse of the type rule specification mechanism and have a great expressiveness and\n    flexibility to specify admitted types in any Python callable entity.\n\n    Type groups are created in the file type_groups.py\n    '

    def __init__(self, grouped_types):
        '\n        Create a new type group that represent the list of types passed as a parameter\n        :param grouped_types: List of types that are included inside this type group\n        :return:\n        '
        self.grouped_types = grouped_types
        type_test.add_type_dict_for_context(locals())


    def __contains__(self, type_):
        '\n        Test if this type group contains the specified type (in operator)\n        :param type_: Type to test\n        :return: bool\n        '
        try:
            return ((type_.get_python_type() in self.grouped_types), type_test.add_type_dict_for_context(locals()))[0]
        except:
            return ((type_ in self.grouped_types), type_test.add_type_dict_for_context(locals()))[0]
        type_test.add_type_dict_for_context(locals())


    def __eq__(self, type_):
        '\n        Test if this type group contains the specified type (== operator)\n        :param type_: Type to test\n        :return: bool\n        '
        try:
            cond1 = (type(type_) in self.grouped_types)
            return (cond1, type_test.add_type_dict_for_context(locals()))[0]
        except:
            return ((type_ in self.grouped_types), type_test.add_type_dict_for_context(locals()))[0]
        type_test.add_type_dict_for_context(locals())


    def __cmp__(self, type_):
        '\n        Test if this type group contains the specified type (compatarion operators)\n        :param type_: Type to test\n        :return: bool\n        '
        try:
            cond1 = (type_.get_python_type() in self.grouped_types)
            cond2 = type_.is_type_instance()
            return ((cond1 and cond2), type_test.add_type_dict_for_context(locals()))[0]
        except:
            return ((type_ in self.grouped_types), type_test.add_type_dict_for_context(locals()))[0]
        type_test.add_type_dict_for_context(locals())


    def __gt__(self, other):
        '\n        Type group sorting. A type group is less than other type group if contains less types or the types contained\n        in the type group are all contained in the other one. Otherwise, is greater than the other type group.\n        :param other: Another type group\n        :return: bool\n        '

        if (len(self.grouped_types) < len(other.grouped_types)):
            return (False, type_test.add_type_dict_for_context(locals()))[0]

        for type_ in self.grouped_types:

            if (type_ not in other.grouped_types):
                return (False, type_test.add_type_dict_for_context(locals()))[0]

        return (True, type_test.add_type_dict_for_context(locals()))[0]
        type_test.add_type_dict_for_context(locals())


    def __lt__(self, other):
        '\n        Type group sorting. A type group is less than other type group if contains less types or the types contained\n        in the type group are all contained in the other one. Otherwise, is greater than the other type group.\n        :param other: Another type group\n        :return: bool\n        '

        if (len(self.grouped_types) > len(other.grouped_types)):
            return (False, type_test.add_type_dict_for_context(locals()))[0]

        for type_ in self.grouped_types:

            if (type_ not in other.grouped_types):
                return (False, type_test.add_type_dict_for_context(locals()))[0]

        return (True, type_test.add_type_dict_for_context(locals()))[0]
        type_test.add_type_dict_for_context(locals())


    def __repr__(self):
        '\n        Textual representation of the type group\n        :return: str\n        '
        ret_str = type(self).__name__
        return (ret_str, type_test.add_type_dict_for_context(locals()))[0]
        type_test.add_type_dict_for_context(locals())


class DependentType:
    "\n    A DependentType is a special base class that indicates that a type group has to be called to obtain the real\n    type it represent. Call is done using the parameters that are trying to match the rule. For example, imagine that\n    we call the + operator with an object that defines the __add__ method and another type to add to. With an object\n    that defines an __add__ method we don't really know what will be the result of calling __add__ over this object\n    with the second parameter, so the __add__ method has to be called (well, in fact, the type inference equivalent\n    version of the __add__ method will be called) to obtain the real return type.\n\n    Dependent types are a powerful mechanism to calculate the return type of operations that depend on calls to\n    certain object members or even to detect incorrect definitions of members in objects (__int__ method defined in\n    object that do not return int, for example).\n    "

    def __init__(self, report_errors=False):
        '\n        Build a Dependent type instance\n        :param report_errors: Flag to indicate if errors found when calling this type will be reported or not (in that\n        case other code will do it)\n        '
        self.report_errors = report_errors
        self.call_arity = 0
        type_test.add_type_dict_for_context(locals())


    def __call__(self, *call_args, **call_kwargs):
        '\n        Call the dependent type. Empty in this implementation, concrete calls must be defined in subclasses\n        '
        pass
        type_test.add_type_dict_for_context(locals())


class HasMember(TypeGroup, DependentType, ):
    '\n        Type of any object that has a member with the specified arity, and that can be called with the corresponding\n        params.\n    '

    def __init__(self, member, expected_return_type, call_arity=0, report_errors=False):
        DependentType.__init__(self, report_errors)
        TypeGroup.__init__(self, [])
        self.member = member
        self.expected_return_type = expected_return_type
        self.member_obj = None
        self.call_arity = call_arity
        type_test.add_type_dict_for_context(locals())


    def format_arity(self):
        str_ = '('
        for i in range(self.call_arity):
            str_ += (('parameter' + str(i)) + ', ')

        if (self.call_arity > 0):
            str_ = str_[:(-2)]

        return ((str_ + ')'), type_test.add_type_dict_for_context(locals()))[0]
        type_test.add_type_dict_for_context(locals())


    def __eq__(self, type_):
        self.member_obj = type_.get_type_of_member(None, self.member)

        if isinstance(self.member_obj, TypeError):

            if (not self.report_errors):
                TypeError.remove_error_msg(self.member_obj)

            return (False, type_test.add_type_dict_for_context(locals()))[0]

        return (True, type_test.add_type_dict_for_context(locals()))[0]
        type_test.add_type_dict_for_context(locals())


    def __call__(self, localization, *call_args, **call_kwargs):

        if callable(self.member_obj.get_python_type()):
            equivalent_type = self.member_obj.invoke(localization, *call_args, **call_kwargs)

            if isinstance(equivalent_type, TypeError):

                if (not self.report_errors):
                    TypeError.remove_error_msg(equivalent_type)

                self.member_obj = None
                return ((False, equivalent_type), type_test.add_type_dict_for_context(locals()))[0]


            if isinstance(self.expected_return_type, UndefinedType):
                self.member_obj = None
                return ((True, equivalent_type), type_test.add_type_dict_for_context(locals()))[0]


            if (self.expected_return_type is DynamicType):
                self.member_obj = None
                return ((True, equivalent_type), type_test.add_type_dict_for_context(locals()))[0]


            if (not issubclass(equivalent_type.get_python_type(), self.expected_return_type)):
                self.member_obj = None
                return ((False, equivalent_type), type_test.add_type_dict_for_context(locals()))[0]
            else:
                return ((True, equivalent_type), type_test.add_type_dict_for_context(locals()))[0]


        self.member_obj = None
        return ((True, None), type_test.add_type_dict_for_context(locals()))[0]
        type_test.add_type_dict_for_context(locals())


    def __repr__(self):
        ret_str = 'Instance defining '
        ret_str += str(self.member)
        ret_str += self.format_arity()
        return (ret_str, type_test.add_type_dict_for_context(locals()))[0]
        type_test.add_type_dict_for_context(locals())

CastsToInt = HasMember('__int__', int, 0)
type_test.add_type_dict_for_main_context(globals())
type_test.generate_type_data_file()