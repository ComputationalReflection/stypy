class UndefinedType():
    """
    The type of an undefined variable
    """

    def __str__(self):
        return 'Undefined'

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return isinstance(other, UndefinedType)


class DynamicType():
    """
    Any type (type cannot be statically calculated)
    """

    def __init__(self, *members):
        TypeGroup.__init__(self, [])
        self.members = members

    def __eq__(self, type_):
        return True


class BaseTypeGroup(object):
    """
    All type groups inherit from this class
    """

    def __str__(self):
        return self.__repr__()


class TypeGroup(BaseTypeGroup):
    """
    A TypeGroup is an entity used in the rule files to group several Python types over a named identity. Type groups
    are collections of types that have something in common, and Python functions and methods usually admits any of them
    as a parameter when one of them is valid. For example, if a Python library function works with an int as the first
    parameter, we can also use bool and long as the first parameter without runtime errors. This is for exameple a
    TypeGroup that will be called Integer

    Not all type groups are defined by collections of Python concrete types. Other groups identify Python objects with
    a common member or structure (Iterable, Overloads__str__ identify any Python object that is iterable and any Python
    object that has defined the __str__ method properly) or even class relationships (SubtypeOf type group only matches
    with classes that are a subtype of the one specified.

    Type groups are the workhorse of the type rule specification mechanism and have a great expressiveness and
    flexibility to specify admitted types in any Python callable entity.

    Type groups are created in the file type_groups.py
    """

    def __init__(self, grouped_types):
        """
        Create a new type group that represent the list of types passed as a parameter
        :param grouped_types: List of types that are included inside this type group
        :return:
        """
        self.grouped_types = grouped_types

    def __contains__(self, type_):
        """
        Test if this type group contains the specified type (in operator)
        :param type_: Type to test
        :return: bool
        """
        # if hasattr(type_, "get_python_type"):
        #     return type_.get_python_type() in self.grouped_types
        #
        # return type_ in self.grouped_types
        try:
            return type_.get_python_type() in self.grouped_types
        except:
            return type_ in self.grouped_types

    def __eq__(self, type_):
        """
        Test if this type group contains the specified type (== operator)
        :param type_: Type to test
        :return: bool
        """
        # if hasattr(type_, "get_python_type"):
        #     return type_.get_python_type() in self.grouped_types
        # return type_ in self.grouped_types
        try:
            cond1 = type(type_) in self.grouped_types

            return cond1
        except:
            return type_ in self.grouped_types

    def __cmp__(self, type_):
        """
        Test if this type group contains the specified type (compatarion operators)
        :param type_: Type to test
        :return: bool
        """
        # if hasattr(type_, "get_python_type"):
        #     return type_.get_python_type() in self.grouped_types
        #
        # return type_ in self.grouped_types
        try:
            # return type_.get_python_type() in self.grouped_types
            cond1 = type(type_) in self.grouped_types

            return cond1
        except:
            return type_ in self.grouped_types

    def __gt__(self, other):
        """
        Type group sorting. A type group is less than other type group if contains less types or the types contained
        in the type group are all contained in the other one. Otherwise, is greater than the other type group.
        :param other: Another type group
        :return: bool
        """
        if len(self.grouped_types) < len(other.grouped_types):
            return False

        for type_ in self.grouped_types:
            if type_ not in other.grouped_types:
                return False

        return True

    def __lt__(self, other):
        """
        Type group sorting. A type group is less than other type group if contains less types or the types contained
        in the type group are all contained in the other one. Otherwise, is greater than the other type group.
        :param other: Another type group
        :return: bool
        """
        if len(self.grouped_types) > len(other.grouped_types):
            return False

        for type_ in self.grouped_types:
            if type_ not in other.grouped_types:
                return False

        return True

    def __repr__(self):
        """
        Textual representation of the type group
        :return: str
        """
        # ret_str = type(self).__name__  + "("
        # for type_ in self.grouped_types:
        #     if hasattr(type_, '__name__'):
        #         ret_str += type_.__name__ + ", "
        #     else:
        #         ret_str += str(type_) + ", "
        #
        # ret_str = ret_str[:-2]
        # ret_str+=")"

        ret_str = type(self).__name__
        return ret_str


class DependentType:
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


class HasMember(TypeGroup, DependentType):
    """
        Type of any object that has a member with the specified arity, and that can be called with the corresponding
        params.
    """

    def __init__(self, member, expected_return_type, call_arity=0, report_errors=False):
        DependentType.__init__(self, report_errors)
        TypeGroup.__init__(self, [])
        self.member = member
        self.expected_return_type = expected_return_type
        self.member_obj = None
        self.call_arity = call_arity

    def format_arity(self):
        str_ = "("
        for i in range(self.call_arity):
            str_ += "parameter" + str(i) + ", "

        if self.call_arity > 0:
            str_ = str_[:-2]

        return str_ + ")"

    def __eq__(self, type_):
        self.member_obj = type_.get_type_of_member(None, self.member)
        if isinstance(self.member_obj, TypeError):
            if not self.report_errors:
                TypeError.remove_error_msg(self.member_obj)
            return False

        return True

    def __call__(self, localization, *call_args, **call_kwargs):
        if callable(self.member_obj.get_python_type()):
            # Call the member
            equivalent_type = self.member_obj.invoke(localization, *call_args, **call_kwargs)

            # Call was impossible: Invokation error has to be removed because we provide a general one later
            if isinstance(equivalent_type, TypeError):
                if not self.report_errors:
                    TypeError.remove_error_msg(equivalent_type)
                self.member_obj = None
                return False, equivalent_type

            # Call was possible, but the expected return type cannot be predetermined (we have to recheck it later)
            if isinstance(self.expected_return_type, UndefinedType):
                self.member_obj = None
                return True, equivalent_type

            # Call was possible, but the expected return type is Any)
            if self.expected_return_type is DynamicType:
                self.member_obj = None
                return True, equivalent_type

            # Call was possible, so we check if the predetermined return type is the same that the one that is returned
            if not issubclass(equivalent_type.get_python_type(), self.expected_return_type):
                self.member_obj = None
                return False, equivalent_type
            else:
                return True, equivalent_type

        self.member_obj = None
        return True, None

    def __repr__(self):
        ret_str = "Instance defining "
        ret_str += str(self.member)
        ret_str += self.format_arity()
        return ret_str


CastsToInt = HasMember("__int__", int, 0)
