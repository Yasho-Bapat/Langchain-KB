from dotenv import load_dotenv

from ask_viridium_ai.routes import MainRoutes

load_dotenv()

from flask import Flask, jsonify, redirect
from flask_cors import CORS
from apispec import APISpec
from flask_swagger_ui import get_swaggerui_blueprint
from apispec.ext.marshmallow import MarshmallowPlugin
from apispec_webframeworks.flask import FlaskPlugin

from global_constants import GlobalConstants

global_constants = GlobalConstants

app = Flask(__name__)
CORS(app)

main_routes = MainRoutes()
app.register_blueprint(main_routes.blueprint, url_prefix=GlobalConstants.api_version)

swagger_endpoint = global_constants.swagger_endpoint
# Swagger and APISpec setup
spec = APISpec(
    title=global_constants.apispec_config.title,
    version=global_constants.apispec_config.version,
    openapi_version=global_constants.apispec_config.openapi_version,
    plugins=[FlaskPlugin(), MarshmallowPlugin()],
)

with app.test_request_context():
    for view in [
        main_routes.home,
        main_routes.ask_viridium_ai,
        main_routes.health_check,
    ]:
        spec.path(view=view)

swaggerui_blueprint = get_swaggerui_blueprint(
    swagger_endpoint,
    global_constants.api_swagger_json,
    config={"app_name": global_constants.swagger_app_name},
)

app.register_blueprint(swaggerui_blueprint, url_prefix=swagger_endpoint)


@app.route('/')
def root():
    return redirect(global_constants.api_version)


@app.route(global_constants.api_swagger_json)
def create_swagger_spec():
    return jsonify(spec.to_dict())


@app.before_request
def log_request_info():
    # logger.info(f"API REQUEST : {request.method} {request.path}")
    pass


@app.after_request
def log_response_info(response):
    # logger.info(f"API RESPONSE : {response.status}")
    return response


# Create 20 threads to process waiting api calls
# threads = threading_tool.create_and_start_threads(
#     process_waiting_api_calls,
#     num_threads=global_constants.no_of_threads,
#     daemon=True,
# )

if __name__ == "__main__":
    # Main thread will serve the api calls
    app.run(
        threaded=True,
        debug=True,
        port=global_constants.flask_app_port,
        host=global_constants.flask_host,
    )
