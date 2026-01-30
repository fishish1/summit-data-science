
import os
import re
import sys
from pathlib import Path

# Add project root to path to allow imports
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from scripts.export_to_onnx import export_gbm, export_nn, EXPORT_PATHS

def patch_missing_onnx():
    models_dir = project_root / "models"
    
    # GBM Models
    gbm_pattern = re.compile(r"price_predictor_gbm_v(\d+)\.pkl")
    nn_pattern = re.compile(r"price_predictor_nn_v(\d+)\.pth")
    
    gbm_runs = []
    nn_runs = []
    
    for f in os.listdir(models_dir):
        m_gbm = gbm_pattern.match(f)
        if m_gbm:
            gbm_runs.append(int(m_gbm.group(1)))
            
        m_nn = nn_pattern.match(f)
        if m_nn:
            nn_runs.append(int(m_nn.group(1)))
            
    print(f"Found {len(gbm_runs)} GBM models and {len(nn_runs)} NN models.")
    
    # Check for missing ONNX files in the first export path (static_dashboard)
    target_dir = EXPORT_PATHS[0]
    
    for run_id in gbm_runs:
        onnx_file = target_dir / f"gbm_v{run_id}.onnx"
        if not onnx_file.exists():
            print(f"⚠️ Missing ONNX for GBM v{run_id}. Generating...")
            try:
                export_gbm(run_id)
            except Exception as e:
                print(f"❌ Failed to export GBM v{run_id}: {e}")
        else:
            print(f"✅ ONNX for GBM v{run_id} exists.")
            
    for run_id in nn_runs:
        onnx_file = target_dir / f"nn_v{run_id}.onnx"
        if not onnx_file.exists():
            print(f"⚠️ Missing ONNX for NN v{run_id}. Generating...")
            try:
                export_nn(run_id)
            except Exception as e:
                print(f"❌ Failed to export NN v{run_id}: {e}")
        else:
            print(f"✅ ONNX for NN v{run_id} exists.")

if __name__ == "__main__":
    patch_missing_onnx()
