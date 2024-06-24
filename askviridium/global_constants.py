import os
import dotenv

dotenv.load_dotenv()


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
        "title": "Ask Viridium AI",
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

    api_response_messages = {
        "success": "Success",
        "accepted": "Accepted",
        "invalid_request_data": "Invalid request data",
        "unauthorized": "Unauthorized access",
        "forbidden": "Forbidden access",
        "not_found": "Resource not found",
        "global_keywords_not_configured": "Global keywords not configured",
        "method_not_allowed": "Method not allowed for the requested resource",
        "conflict": "Conflict with current state of the resource",
        "internal_server_error": "Internal server error occurred",
        "service_unavailable": "Service temporarily unavailable",
        "server_is_running": "Ask Viridium AI Service is running",
        "missing_required_parameters": "Missing required parameters",
        "error_while_processing_file": "Error while processing file",
    }

    api_response_messages = DotAccessDict(api_response_messages)

    chemical_composition_example = {
        "product_name": "TRIM TC 184B",
        "chemicals": [{"name": "Severely Hydrotreated Petroleum Oil", "cas_no": "64742-65-0",
                       "source": "https://www1.mscdirect.com/MSDS/MSDS00007/01790583-20110708.PDF"}],
        "confidence": 0.85
    }

    analysis_example = {
        "analyzed_material": "0652-W Nylon/ 30655-W nylon with CPT Sealant",
        "composition": "Nylon, CPT Sealant",
        "analysis_method": "Literature review, trade name association",
        "decision": "PFAS (No)",
        "confidence_score": 0.90,
        "primary_reason": "Nylon is a polymer that does not contain PFAS. CPT Sealant does not typically contain PFAS based on available information.",
        "secondary_reason": None,
        "evidence": ["Trade name association with nylon, which is a non-PFAS material",
                     "Lack of information suggesting PFAS presence in CPT Sealant"],
        "health_problems": ["Could lead to asphyxia", "Linked to cancer"],
        "confidence_level": "High",
        "recommendation": "No further investigation is needed as the analyzed materials are not expected to contain PFAS.",
        "suggestion": None,
        "limitations_and_uncertainties": None
    }

    model_name = os.getenv("AZURE_MODEL_NAME")
    deployment_name = os.getenv("AZURE_DEPLOYMENT_NAME")

