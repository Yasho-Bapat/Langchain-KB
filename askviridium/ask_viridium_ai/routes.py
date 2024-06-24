from flask import Blueprint, jsonify, render_template, request

from global_constants import GlobalConstants
from .ask_viridium_ai import AskViridium
from .constants import AskViridiumConstants


class MainRoutes:
    def __init__(self):
        self.blueprint = Blueprint("main_routes", __name__, template_folder="templates", static_folder="static")
        # self.logger = logger
        self.global_constants = GlobalConstants
        self.constants = AskViridiumConstants

        self.blueprint.add_url_rule(
            "/",
            view_func=self.home,
            methods=[self.global_constants.rest_api_methods.get_api],
        )

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

    def home(self):
        return render_template("pages/index.html")

    def ask_viridium_ai(self):
        request_data = request.get_json()
        required_params = [
            self.constants.input_parameters["material_name"],
        ]

        valid_request, missing_params = self.validate_request_data(
            request_data, required_params
        )
        if not valid_request:
            return self.return_api_response(
                self.global_constants.api_status_codes.bad_request,
                self.global_constants.api_response_messages.missing_required_parameters,
                f"{self.global_constants.api_response_parameters.missing_parameters}: {missing_params}",
            )

        ask_vai = AskViridium()
        ask_vai.query(request_data[self.constants.input_parameters["material_name"]], request_data[self.constants.input_parameters["manufacturer_name"]], request_data[self.constants.input_parameters["work_content"]])

        return self.return_api_response(
            self.global_constants.api_status_codes.ok,
            self.global_constants.api_response_messages.success,
            ask_vai.result,
        )

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
