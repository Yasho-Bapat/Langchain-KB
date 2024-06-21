import os


class DotAccessDict(dict):
    def __getattr__(self, attr):
        if attr in self:
            return self[attr]
        else:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{attr}'"
            )


class GlobalConstants(DotAccessDict):
    api_version = "/v1"
    utf_8 = "utf-8"

    flask_host = "0.0.0.0"
    flask_app_port = os.getenv("WEBSITES_PORT", 8000)
    u = "u"
    no_of_threads = int(os.getenv("NoOfThreads", 20))
    api_swagger_json = "/api/swagger.json"
    swagger_app_name = "Ask Viridium AI"
    swagger_endpoint = os.getenv("SwaggerEndpoint", "/api/docs")

    azure_deployment_name = "AZURE_CLIENT_SECRET"
    azure_enpoint = "AZURE_TENANT_ID"

    swagger_config = {
        "app_name": "Keyword Analysis API",  # Set the title of your API
        "docExpansion": "none",  # Controls the default expansion setting for the operations and tags
        "displayOperationId": True,  # Controls the display of operation Ids in operations list
        "displayRequestDuration": True,  # Controls the display of the request duration in the response
        "defaultModelsExpandDepth": 0,  # The default expansion depth for the model on the model-example section
        "defaultModelExpandDepth": 1,  # The default expansion depth for the model on the model section
    }

    api_response_parameters = {
        "status": "status",
        "message": "message",
        "result": "result",
        "identifier": "identifier",
        "id": "id",
        "status_code": "status_code",
        "missing_parameters": "Missing parameters",
        "reason": "reason",
    }

    api_response_parameters = DotAccessDict(api_response_parameters)

    rest_api_methods = {
        "post": "POST",
        "get_api": "GET",  # Using "get_api" because "get" is reserved keyword
        "put": "PUT",
        "delete": "DELETE",
        "patch": "PATCH",
    }

    rest_api_methods = DotAccessDict(rest_api_methods)

    apispec_config = {
        "title": "Keyword Analysis API",
        "version": "1.0.0",
        "openapi_version": "3.0.2",
    }

    apispec_config = DotAccessDict(apispec_config)

    api_status_codes = {
        "ok": 200,
        "created": 201,
        "no_content": 204,
        "bad_request": 400,
        "unauthorized": 401,
        "forbidden": 403,
        "not_found": 404,
        "method_not_allowed": 405,
        "conflict": 409,
        "internal_server_error": 500,
        "service_unavailable": 503,
        "rate_limit_exceeded": 429,
    }
    api_status_codes = DotAccessDict(api_status_codes)