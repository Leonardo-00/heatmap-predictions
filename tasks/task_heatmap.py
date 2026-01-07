''' Snap4city Computing HEATMAP - ClearML Entrypoint.
    Copyright (C) 2026 DISIT Lab http://www.disit.org - University of Florence
'''

import os
import ast
import json
import logging
import traceback
from clearml import Task
from logic.heatmap.heatmap import generate_heatmap

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def normalize_params(params: dict) -> dict:
    """
    Ensures that stringified structures (lists, dicts) from ClearML 
    are converted back to Python objects.
    """
    for k, v in params.items():
        if isinstance(v, str):
            v_stripped = v.strip()
            if v_stripped.startswith(('[', '{', '(')):
                try:
                    params[k] = ast.literal_eval(v_stripped)
                except Exception:
                    pass
    return params

def upload_artifact_safe(task, name, obj):
    """Safe artifact upload for ClearML tracking."""
    try:
        task.upload_artifact(name=name, artifact_object=obj)
        print(f"[OK] Artifact '{name}' uploaded to ClearML.")
    except Exception as e:
        print(f"[WARN] Failed to upload artifact '{name}': {e}")

def run_heatmap_task():
    print("=== Starting Snap4City Heatmap Task on ClearML ===")

    # Initialize Task (ClearML handles project/name when cloned)
    task = Task.init(
        project_name=None, 
        task_name=None, 
        reuse_last_task_id=True
    )
    cl_logger = task.get_logger()

    try:
        # 1. Parameter Acquisition
        raw_params = task.get_parameters_as_dict(cast=True)
        # ClearML structures parameters under a group (usually 'General')
        params = normalize_params(raw_params.get("General", raw_params))
        
        print("Parameters received and normalized:")
        cl_logger.report_text(json.dumps(params, indent=2))
        
        # 2. Heatmap Generation
        # Now returns the HeatmapStatus dictionary directly
        result = generate_heatmap(params)
        
        # 3. Logging Results
        print("Generation Result:")
        print(json.dumps(result, indent=2))

        # 4. Artifact Management
        # The 'result' is the actual heatmap info dictionary
        upload_artifact_safe(task, "heatmap_execution_summary", result)

        # Handle specific artifacts if generated (e.g., local logs)
        if os.path.exists("uvicorn.log"):
            upload_artifact_safe(task, "execution_logs", "uvicorn.log")

        # 5. Final Status
        if "message" in result and any("ERROR" in m or "failed" in m.lower() for m in result["message"]):
            cl_logger.report_text("Task finished with internal errors (check summary).")
        else:
            cl_logger.report_text("Task completed successfully.")

    except Exception:
        error_msg = traceback.format_exc()
        print(error_msg)
        cl_logger.report_text(error_msg)
        raise

if __name__ == "__main__":
    run_heatmap_task()