import json
import traceback
from clearml import Task
import logging
from logic.heatmap.heatmap import generate_heatmap

def setup_logging(verbose: bool):

    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


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

        setup_logging(params.get("verbose", False))
        result = generate_heatmap(params)
        
        logger.report_text(json.dumps(result, indent=2))

        upload_artifact_safe(task, "heatmap_info", result)

        logger.report_text("Task completato con successo")

    except Exception:
        logger.report_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    run_heatmap_task()