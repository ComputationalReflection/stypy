def generators():
    def f(x):
        if False:
            return str(x)
        return x

    r = [f(x) for x in range(10)]
    r2 = r[0].capitalize()  # Unreported, runtime crash


generators()
