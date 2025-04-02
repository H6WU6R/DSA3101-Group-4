from flask import Flask, jsonify, request
import json
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

@app.route('/run_module/<module_name>', methods=['POST'])
def run_module(module_name):
    module_map = {
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
    
    try:
        if module_name in module_map:
            result = module_map[module_name]()
            return jsonify({
                'status': 'success',
                'module': module_name,
                'message': f'Successfully executed module {module_name}',
                'result': str(result) if result is not None else None
            })
        else:
            return jsonify({
                'status': 'error',
                'message': f'Module {module_name} not found'
            }), 404
            
    except Exception as e:
        return jsonify({
            'status': 'error',
            'module': module_name,
            'message': str(e)
        }), 500

@app.route('/run_all', methods=['POST'])
def run_all():
    results = {}
    try:
        for module_name, module_func in {
            'A1': A1_main,
            'A2': A2_main,
            'A3': A3_main,
            'A4': A4_main,
            'A5': A5_app,
            'B1': B1_main,
            'B3': B3_main,
            'B4': B4_main,
            'B5': B5_main
        }.items():
            try:
                result = module_func()
                results[module_name] = {
                    'status': 'success',
                    'result': str(result) if result is not None else None
                }
            except Exception as e:
                results[module_name] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        return jsonify({
            'status': 'success',
            'results': results
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)