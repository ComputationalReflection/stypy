
# function: function
# function() -> Simple instance 

def function():
    # Simple: Simple

    class Simple:
        # self: instance
        sample_att = 3
        # <Dead code detected>

        def sample_method(self):
            self.att = 'sample'

    return Simple()

# ret: Simple instance
ret = function()