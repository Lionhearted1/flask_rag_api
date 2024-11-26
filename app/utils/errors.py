class AppError(Exception):
    def __init__(self, message, status_code=500, payload=None):
        super().__init__()
        self.message = message
        self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['message'] = self.message
        return rv

class DocumentProcessingError(AppError):
    def __init__(self, message, payload=None):
        super().__init__(message, status_code=400, payload=payload)

class VectorStorageError(AppError):
    def __init__(self, message, payload=None):
        super().__init__(message, status_code=500, payload=payload)

class ChatProcessingError(AppError):
    def __init__(self, message, payload=None):
        super().__init__(message, status_code=500, payload=payload)
