from flask import Flask, jsonify, request, abort
import logging

# Import module entry points
from A1_Customer_Segmentation.A1_main import main as A1_main
from A2_Customer_Engagement.A2_main import main as A2_main
from A3_Behavioral_Patterns_Analysis.A3_main import main as A3_main
from A4_Campaign_Impact_Analysis.A4_main import main as A4_main
from A5_Segmentation_Updates.app import main as A5_app
from B1_Predicting_Customer_Preferences.B1_main import main as B1_main
from B3_Campaign_ROI_Evaluation.B3_main import main as B3_main
from B4_Cost_Effectiveness_Of_Campaigns.B4_main import main as B4_main
from B5_Customer_Retention_Strategies.B5_main import main as B5_main

app = Flask(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# Mapping for modules
MODULE_MAP = {
    'A1': A1_main,
    'A2': A2_main,
    'A3': A3_main,
    'A4': A4_main,
    'A5': A5_app,
    'B1': B1_main,
    'B3': B3_main,
    'B4': B4_main,
    'B5': B5_main
}

@app.route('/', methods=['GET'])
def index():
    """Root endpoint providing API information."""
    return jsonify({
        'status': 'success',
        'message': 'API is running',
        'available_endpoints': {
            'GET /modules': 'List available modules',
            'POST /run_module/<module_name>': 'Run a specific module',
            'POST /run_all': 'Run all modules'
        }
    })

@app.route('/modules', methods=['GET'])
def list_modules():
    """Endpoint to list all available module keys."""
    return jsonify({
        'status': 'success',
        'modules': list(MODULE_MAP.keys())
    })

@app.route('/run_module/<module_name>', methods=['POST'])
def run_module(module_name):
    """Endpoint to run a specific module."""
    if module_name not in MODULE_MAP:
        app.logger.error("Module %s not found", module_name)
        abort(404, description=f"Module {module_name} not found")
    try:
        result = MODULE_MAP[module_name]()
        return jsonify({
            'status': 'success',
            'module': module_name,
            'message': f'Successfully executed module {module_name}',
            'result': str(result) if result is not None else None
        }), 200
    except Exception as e:
        app.logger.exception("Error executing module %s", module_name)
        abort(500, description=str(e))

@app.route('/run_all', methods=['POST'])
def run_all():
    """Endpoint to run all modules sequentially."""
    results = {}
    for module_name, module_func in MODULE_MAP.items():
        try:
            result = module_func()
            results[module_name] = {
                'status': 'success',
                'result': str(result) if result is not None else None
            }
        except Exception as e:
            app.logger.exception("Error executing module %s", module_name)
            results[module_name] = {
                'status': 'error',
                'error': str(e)
            }
    return jsonify({
        'status': 'success',
        'results': results
    }), 200

# Custom error handlers
@app.errorhandler(404)
def resource_not_found(e):
    return jsonify({'status': 'error', 'message': str(e)}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'status': 'error', 'message': 'Internal Server Error: ' + str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)