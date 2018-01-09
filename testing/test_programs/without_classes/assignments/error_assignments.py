def fun():
    if True:
        return 3
    else:
        return [3, 4]


fun()[0] = 4  # Not Reported
