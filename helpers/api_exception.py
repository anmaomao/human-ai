class ApiException(Exception):
    code = 400

    def __init__(self, message, status_code=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.code = status_code

    def __str__(self):
        return 'ApiException: ' + self.message
