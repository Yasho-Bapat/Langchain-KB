class MaxProcessingTimeExceededException(Exception):
    def __init__(self, message="Max processing time limit reached!", details=None):
        super().__init__(message, details)
