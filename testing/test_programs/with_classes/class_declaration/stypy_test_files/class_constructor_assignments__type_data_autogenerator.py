
from stypy.code_generation.type_inference_programs.checking.type_data_file_writer import TypeDataFileWriter

type_test = TypeDataFileWriter(__file__)

class Record:

    def __init__(self, PtrComp=None, Discr=0, EnumComp=0, IntComp=0, StringComp=0):
        self.PtrComp = PtrComp
        self.Discr = Discr
        self.EnumComp = EnumComp
        self.IntComp = IntComp
        self.StringComp = StringComp
        type_test.add_type_dict_for_context(locals())


    def copy(self):
        return (Record(self.PtrComp, self.Discr, self.EnumComp, self.IntComp, self.StringComp), type_test.add_type_dict_for_context(locals()))[0]
        type_test.add_type_dict_for_context(locals())

r = Record()
x1 = r.PtrComp
x2 = r.Discr
x3 = r.EnumComp
x4 = r.IntComp
x5 = r.StringComp
r2 = r.copy()
y1 = r2.PtrComp
y2 = r2.Discr
y3 = r2.EnumComp
y4 = r2.IntComp
y5 = r2.StringComp
type_test.add_type_dict_for_main_context(globals())
type_test.generate_type_data_file()