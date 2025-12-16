import os
import json
import traceback
from clearml import Task
from logic.heatmap.heatmap import generate_heatmap

def upload_artifact_safe(task, name, obj):
    """Upload sicuro con messaggi più chiari."""
    try:
        task.upload_artifact(name=name, artifact_object=obj)
        print(f"[OK] Artifact '{name}' caricato.")
    except Exception as e:
        print(f"[WARN] Impossibile caricare artifact '{name}': {e}")


def run_heatmap_task():
    print("=== Avvio task heatmap ===")

    task = Task.init(
        project_name=None,
        task_name=None,
        reuse_last_task_id=True
    )
    logger = task.get_logger()

    try:
        params = task.get_parameters_as_dict(cast=True)
        params = params["General"]
        
        print("Parametri ricevuti:")
        logger.report_text(json.dumps(params, indent=2))

        result = generate_heatmap(params)
        
        print(result)

        upload_artifact_safe(task, "heatmap_info", result.get("info_heatmap", {}))

        for key in ["output_file_path", "interpolated_csv"]:
            path = result.get(key)
            if path and os.path.exists(path):
                upload_artifact_safe(task, key, path)

        logger.report_text("Task completato con successo")

    except Exception:
        logger.report_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    run_heatmap_task()
