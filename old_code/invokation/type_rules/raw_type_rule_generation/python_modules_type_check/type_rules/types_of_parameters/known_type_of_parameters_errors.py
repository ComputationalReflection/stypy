"""
Known type errors to filter error TypeRules when applying the "type polling" method to guess admitted parameter types
in a function or method call.

NOTE: Converted from lambda function list for performance reasons.
"""

def is_known_type_error(msg):
    if ("bad operand type" in msg or
            ("can't convert" in msg and "to" in msg) or
            ("a" in msg and "is required" in msg) or
                "unsupported" in msg or
            ("must be" in msg and "not" in msg) or
                "is not callable" in msg or
                "expected sequence" in msg or
                "bad argument type" in msg or
            ("requires" in msg and "but received" in msg) or
            ("cannot concatenate" in msg and "and" in msg and "objects" in msg) or
            ("requires" in msg and "as left operand" in msg and "not" in msg) or
                "expected string or buffer" in msg or
            ("unhashable type" in msg and "is not iterable" in msg) or
                "object does not support item" in msg or
                "iteration over non-sequence" in msg or
                "need string or buffer" in msg or
                "object can't be" in msg or
                "argument expected, got" in msg or
                "object doesn't support" in msg or
                "object does not support" in msg or
                "can only concatenate" in msg or
            ("can't multiply" in msg and "by non-int of type" in msg) or
                "index must be integer" in msg or
                "object cannot be interpreted as an index" in msg or
                "buffer is read-only" in msg or
                "object is unsliceable" in msg or
                "indices must be integers" in msg or
                "no ordering relation is defined for" in msg or
            ("must assign" in msg and "to" in msg) or
                "does not have the buffer interface" in msg or
                "locals must be a mapping" in msg or
                "attribute name must be" in msg or
                "must support iteration" in msg or
                "locals must be a mapping" in msg or
            ("expected" in msg and "found" in msg) or
                "is not a type object" in msg or
            ("can't pickle" in msg and "objects" in msg) or
                "can only concatenate" in msg or
                "is not a dictionary" in msg or
            ("expected" in msg and "argument" in msg) or

            ("equired argument" in msg) or
            ("equires at least" in msg) or
            ("takes at least" in msg) or
            ("takes at most" in msg) or
            ("takes exactly" in msg) or
            ("takes" in msg and "arguments" in msg) or
            ("needs" in msg and "argument" in msg) or
            ("needs" in msg and "args" in msg) or
            ("expected" in msg and "arguments" in msg) or
            ("requires" in msg and "but receives" in msg) or
            ("requires" in msg and "but received" in msg) or
            ("takes no parameters" in msg) or
            ("takes no arguments" in msg)
    ):
        return True
    return False

if __name__ == '__main__':
    print is_known_type_error("StopIteration")

