class VisionException(Exception):
    def __init__(self, *args):
        super(VisionException, self).__init__(*args)
        self.message = self.args[0]
    def __str__(self):
        return 'VisionException: ' + self.message
