import sys

__member_lists = {
    "__builtins__": filter(lambda member: not (member == "print"
                                      or member == "None"
                                      or member == "__debug__"
                                      or member == "Ellipsis"
                                      or member == "NotImplemented"),
                         dir(sys.modules["__builtin__"])
                    ),
    #list: ['__getattribute__', '__new__', '__format__', '__setslice__', '__delattr__']
}

def get_members_of_object(obj):
    """
    Extract the usable members of any python object
    :param obj: Any python object
    :return: The result of calling dir(obj)
    """
    try:
        if obj is sys.modules["__builtin__"]:
            return __member_lists["__builtins__"]

        if obj in __member_lists:
            return __member_lists[obj]
    except Exception:
        pass

    return dir(obj)


# import sys
#
# print dir(__builtins__)
# print get_members_of_object(sys.modules["__builtin__"])