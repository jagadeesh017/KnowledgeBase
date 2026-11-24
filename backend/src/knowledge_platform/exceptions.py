class KnowledgePlatformError(Exception):
    pass


class DocumentParseError(KnowledgePlatformError):
    pass


class EmptyDocumentError(KnowledgePlatformError):
    pass


class ServiceUnavailableError(KnowledgePlatformError):
    pass
