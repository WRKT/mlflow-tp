import mlflow
from ultralytics import YOLO


def train_demo():
    with mlflow.start_run(run_name="demo", log_system_metrics=True) as run:
        model = YOLO("yolo11n.pt")
        model.train(data="coco8.yaml", epochs=2)
        mlflow.log_artifact(
            local_path="requirements.txt",
            artifact_path="environment",
            run_id=run.info.run_id,
        )


if __name__ == "__main__":
    train_demo()