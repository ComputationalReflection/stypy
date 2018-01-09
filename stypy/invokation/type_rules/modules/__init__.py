"""
Directory in which all type rules and type modifiers are stored. They are grouped by folders (one per module, one folder
per module class) and type rule and type modifier files for members are distinguished by the member name, using
 a predefined format to assign names to these files. The format of the files is the following:

 Type rule files:

 - It contains a dictionary called type_rules_of_members
 - Each dictionary element has the following structure:

 <member name>: [
    ((<parameter1 type>, <parameter2 type>, ...,), <return type>),
    ...
 ]

 Parameter and return types may be Python types or Type Groups (see the corresponding file to learn about them). For
 example:

     'bytearray': [
        ((), bytearray),
        ((IterableDataStructureWithTypedElements(Integer, Overloads__trunc__),), bytearray),
        ((Integer,), bytearray),
        ((Str,), bytearray),
    ],

    Means that the function 'bytearray' admits 0 or 1 parameters. In case a parameter is passed it accepts the type
    groups Integer (comprising byte, int and long) or Str(composed by str or unicode). It also accepts the type
    group (IterableDataStructureWithTypedElements(Integer, Overloads__trunc__), which represent iterable object whose
    stored types matches the ones specified (Integer, entity that overloads the method __trunc__)).

  Type modifier files:

  - Has a class names TypeModifiers
  - For every method that needs a type modifier, this class has a static method with the same name of the method
  - The type modifier methods follows the following signature:

    @staticmethod
    def set(localization, proxy_obj, arguments):

    Thus receiving The type inference proxy that hold the called entity and the arguments of the call

"""
__author__ = 'Redondo'
