def problematic_get():
    if True:
        return "hi"
    else:
        return [1, 2]


x = problematic_get() / 3
