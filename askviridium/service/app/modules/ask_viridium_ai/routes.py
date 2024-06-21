from flask import Blueprint, jsonify, request

from askviridium.service.app.global_constants import GlobalConstants


class MainRoutes:
    def __init__(self):
        self.blueprint = Blueprint("main_routes", __name__)
        # self.logger = logger
        self.global_constants = GlobalConstants

        self.blueprint.add_url_rule(
            "/ask-viridium-ai",
            view_func=self.ask_viridium_ai,
            methods=[self.global_constants.rest_api_methods.post],
        )

        self.blueprint.add_url_rule("/health", view_func=self.health_check)


    def return_api_response(self, status, message, result=None, additional_data=None):
        response_data = {
            self.global_constants.api_response_parameters.status: status,
            self.global_constants.api_response_parameters.message: message,
            self.global_constants.api_response_parameters.result: result,
        }
        if additional_data:
            response_data.update(additional_data)

        return jsonify(response_data), status

    def validate_request_data(self, request_data, required_params):
        missing_params = [
            param for param in required_params if param not in request_data
        ]
        if missing_params:
            return False, missing_params
        return True, None

    def ask_viridium_ai(self):
        return "Hello World!"

    def health_check(self):
        """
        ---
        get:
          summary: Health check
          responses:
            200:
              description: Server is running
        """
        return self.return_api_response(
            self.global_constants.api_status_codes.ok,
            self.global_constants.api_response_messages.server_is_running,
        )

