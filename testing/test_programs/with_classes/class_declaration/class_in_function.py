
def function():
    class Simple:
        sample_att = 3

        def sample_method(self):
            self.att = "sample"

    return Simple()

ret = function()


