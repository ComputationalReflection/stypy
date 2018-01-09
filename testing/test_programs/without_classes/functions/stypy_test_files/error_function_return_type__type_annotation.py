
# problematic_get: function
# problematic_get() -> str \/ list[int] 

def problematic_get():

    if True:
        return 'hi'
    else:
        return [1, 2]


# x: TypeError
x = (problematic_get() / 3)