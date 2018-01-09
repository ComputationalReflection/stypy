
import numpy

from stypy.invokation.type_rules.type_groups.type_group_generator import *

type_rules_of_members = {
    'fit': [
        ((IterableDataStructure, IterableDataStructure), DynamicType),
        ((IterableDataStructure, IterableDataStructure, Integer), DynamicType),
        ((IterableDataStructure, IterableDataStructure, IterableDataStructure), DynamicType),
        ((IterableDataStructure, IterableDataStructure, IterableDataStructure, AnyType), DynamicType),
        ((IterableDataStructure, IterableDataStructure, IterableDataStructure, AnyType, AnyType), DynamicType),
        ((IterableDataStructure, IterableDataStructure, IterableDataStructure, AnyType, AnyType, AnyType), DynamicType),
        ((IterableDataStructure, IterableDataStructure, IterableDataStructure, AnyType, AnyType, AnyType, AnyType), DynamicType),
        ((IterableDataStructure, IterableDataStructure, IterableDataStructure, AnyType, AnyType, AnyType, AnyType, AnyType), DynamicType),

    ],
}