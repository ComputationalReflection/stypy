def function(x, **kwargs):
    a = 0
    if a > 0:
        return int(x)
    else:
        return kwargs[0]  # Should warn about None


y = function(3, val="hi")

y2 = y.thisdonotexist()  # Unreported
