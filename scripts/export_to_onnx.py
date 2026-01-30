import torch
import joblib
import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import numpy as np
import os
import json
import io
import argparse
from summit_housing.ml.nn import PriceNet
from pathlib import Path

def get_export_paths():
    """
    Determine where to export ONNX model files.
    """
    paths = []
    project_root = Path(__file__).resolve().parent.parent
    
    # 1. Internal Static Dashboard (Guaranteed)
    internal_path = project_root / "static_dashboard/models"
    internal_path.mkdir(parents=True, exist_ok=True)
    paths.append(internal_path)
    
    # 2. Sibling Repo (website)
    sibling_path = project_root.parent / "brian.fishman.info/public/projects/summit/models"
    if sibling_path.exists():
        paths.append(sibling_path)
    
    return paths

EXPORT_PATHS = get_export_paths()

def get_champion_run_id(model_type, default_id):
    """
    Retrieves the run_id for the current champion model from the registry.
    """
    try:
        with open("models/champion_registry.json", "r") as f:
            registry = json.load(f)
        
        # Map internal types to registry keys
        key_map = {
            'gbm': 'gbm',
            'nn': 'price_net_macro'
        }
        reg_key = key_map.get(model_type, model_type)
        
        if reg_key in registry:
            run_id = registry[reg_key]['run_id']
            print(f"🏆 Found champion {model_type} (run_id: {run_id})")
            return run_id
            
    except FileNotFoundError:
        print("⚠️ Champion registry not found.")
    except Exception as e:
        print(f"⚠️ Error reading registry: {e}")
    
    print(f"⚠️ Using fallback run_id: {default_id}")
    return default_id

def export_gbm(run_id=None):
    if run_id is None:
        run_id = get_champion_run_id('gbm', 4)
    
    print(f"Exporting GBM model v{run_id}...")
    
    model_path = f"models/price_predictor_gbm_v{run_id}.pkl"
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return

    pipeline = joblib.load(model_path)
    preprocessor = pipeline.named_steps['preprocessor']
    regressor_wrapper = pipeline.named_steps['model']
    actual_model = regressor_wrapper.regressor_
    
    input_dim = preprocessor.named_transformers_['num'].get_feature_names_out().shape[0] + \
                preprocessor.named_transformers_['cat'].get_feature_names_out().shape[0]
                
    initial_type = [('input', FloatTensorType([None, input_dim]))]
    onx = convert_sklearn(actual_model, initial_types=initial_type, target_opset=12)
    
    for target_dir in EXPORT_PATHS:
        os.makedirs(target_dir, exist_ok=True)
        
        files_to_write = [f"gbm_v{run_id}.onnx"]
        
        # Check if this run is the champion
        champ_id = get_champion_run_id('gbm', -1)
        if run_id == champ_id:
             files_to_write.append("gbm.onnx")

        for fname in files_to_write:
            with open(os.path.join(target_dir, fname), "wb") as f:
                f.write(onx.SerializeToString())
            
        scaler = preprocessor.named_transformers_['num']
        ohe = preprocessor.named_transformers_['cat']
        
        metadata = {
            "num_means": scaler.mean_.tolist(),
            "num_scales": scaler.scale_.tolist(),
            "cat_categories": [cat.tolist() for cat in ohe.categories_],
            "input_features": {
                "numeric": scaler.feature_names_in_.tolist(),
                "categorical": ohe.feature_names_in_.tolist()
            },
            "run_id": run_id
        }
        
        meta_files = [f"gbm_metadata_v{run_id}.json"]
        if run_id == champ_id:
            meta_files.append("gbm_metadata.json")
            
        for fname in meta_files:
            with open(os.path.join(target_dir, fname), "w") as f:
                json.dump(metadata, f)
                
    print(f"✅ GBM v{run_id} exported successfully")

def export_nn(run_id=None):
    if run_id is None:
        run_id = get_champion_run_id('nn', 12)
        
    print(f"Exporting NN model v{run_id}...")
    
    # Load params from history
    try:
        with open("models/experiment_history.json", "r") as f:
            history = json.load(f)
            run_data = next((r for r in history if r['run_id'] == run_id), None)
            if not run_data:
                print(f"❌ Run data not found in history for run_id {run_id}")
                return
            params = run_data['parameters']
    except Exception as e:
        print(f"❌ Error loading experiment history: {e}")
        return
    
    preprocessor_path = f"models/nn_preprocessor_v{run_id}.pkl"
    model_path = f"models/price_predictor_nn_v{run_id}.pth"
    y_scaler_path = f"models/nn_y_scaler_v{run_id}.pkl"
    
    if not os.path.exists(preprocessor_path) or not os.path.exists(model_path):
        print(f"❌ Model artifacts missing for run {run_id}")
        return

    preprocessor = joblib.load(preprocessor_path)
    input_dim = preprocessor.named_transformers_['num'].get_feature_names_out().shape[0] + \
                preprocessor.named_transformers_['cat'].get_feature_names_out().shape[0]
                
    model = PriceNet(input_dim, params=params)
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.eval()
    
    dummy_input = torch.randn(1, input_dim)
    
    # Export to buffer to ensure we can load/save without external data issues
    f_buf = io.BytesIO()
    torch.onnx.export(model, dummy_input, f_buf, 
                      input_names=['input'], output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
    
    f_buf.seek(0)
    onnx_model = onnx.load(f_buf)
    
    for target_dir in EXPORT_PATHS:
        os.makedirs(target_dir, exist_ok=True)
        
        files_to_write = [f"nn_v{run_id}.onnx"]
        champ_id = get_champion_run_id('nn', -1)
        if run_id == champ_id:
             files_to_write.append("nn.onnx")

        for fname in files_to_write:
            onx_path = os.path.join(target_dir, fname)
            onnx.save_model(onnx_model, onx_path, save_as_external_data=False)
            
            # Verify no .data file exists in target_dir
            data_file = onx_path + ".data"
            if os.path.exists(data_file):
                print(f"Warning: external data file created, deleting: {data_file}")
                os.remove(data_file)
        
        scaler = preprocessor.named_transformers_['num']
        ohe = preprocessor.named_transformers_['cat']
        y_scaler = joblib.load(y_scaler_path)
        
        metadata = {
            "num_means": scaler.mean_.tolist(),
            "num_scales": scaler.scale_.tolist(),
            "cat_categories": [cat.tolist() for cat in ohe.categories_],
            "y_mean": y_scaler.mean_.tolist(),
            "y_scale": y_scaler.scale_.tolist(),
            "input_features": {
                "numeric": scaler.feature_names_in_.tolist(),
                "categorical": ohe.feature_names_in_.tolist()
            },
            "run_id": run_id
        }
        
        meta_files = [f"nn_metadata_v{run_id}.json"]
        if run_id == champ_id:
            meta_files.append("nn_metadata.json")
            
        for fname in meta_files:
            with open(os.path.join(target_dir, fname), "w") as f:
                json.dump(metadata, f)
                
    print(f"✅ NN v{run_id} exported successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export models to ONNX")
    parser.add_argument("--run-id", type=int, help="Specific Run ID to export (optional)")
    parser.add_argument("--type", type=str, choices=['gbm', 'nn'], help="Model type to export (optional)")
    
    args = parser.parse_args()
    
    print(f"📊 Running ONNX Export to {len(EXPORT_PATHS)} destinations...")
    
    if args.run_id:
        if args.type == 'gbm':
            export_gbm(args.run_id)
        elif args.type == 'nn':
            export_nn(args.run_id)
        else:
             # Try both or assume based on run?
             print(f"Attempting to export both types for run {args.run_id}...")
             export_gbm(args.run_id)
             export_nn(args.run_id)
    else:
        # Default behavior: Export Champions
        export_gbm()
        export_nn()
