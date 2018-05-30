import re

class ASF:

    def __init__(self):
        self.version = None
        self.name = None
        self.units = None
        self.documentation = None
        self.root = None
        self.bones = None

    def load_from_file(filename):

        asf = ASF()

        with open(filename) as f:
            f_contents = "".join(f.readlines())

        asf.version = re.search(r":version\s+(\S+)").group(1)
        asf.name = re.search(r":version\s+(\S+)").group(1)
        # TODO Implement parsing units section
