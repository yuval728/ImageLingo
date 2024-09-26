import torch
import mlflow
import argparse
import warnings
warnings.filterwarnings("ignore")

def get_models_from_artifact_checkpoint(run_id, is_best_artifact):
    mlflow_client = mlflow.tracking.MlflowClient()
    artifacts = mlflow_client.list_artifacts(run_id)
    for artifact in artifacts:
        if 'checkpoint' in artifact.path:
            if is_best_artifact and 'BEST' not in artifact.path:
                continue
            elif not is_best_artifact and 'BEST' in artifact.path:
                continue
            checkpoint_path = mlflow_client.download_artifacts(run_id, artifact.path)
            print(f"Checkpoint path: {checkpoint_path}")
            return torch.load(checkpoint_path)
    
    raise Exception(f"Artifact not found in run {run_id}")
    

def register_model(run_id, is_best_artifact, model_type, model_name):
   
    if model_type not in ["encoder", "decoder"]:
        raise Exception("Model type must be either 'encoder' or 'decoder'")
    
    with mlflow.start_run(run_id=run_id, nested=True):
        checkpoint = get_models_from_artifact_checkpoint(run_id, is_best_artifact)
        
        model = checkpoint[model_type]
        
        mlflow.pytorch.log_model(model, model_name)
            
        mlflow.register_model(f"runs:/{run_id}/{model_name}", model_name)

        
        
    
def parse_args():
    parser = argparse.ArgumentParser(description="Register models")
    parser.add_argument("--run_id", type=str, help="Run ID of the model to register", required=True)
    # parser.add_argument("--artifact_name", type=str, help="Artifact name of the model to register", required=True)
    parser.add_argument("--best_artifact", action="store_true", help="Register the best artifact", default=False)
    parser.add_argument("--tracking_uri", type=str, help="Tracking URI of the MLflow server", default="http://localhost:5000")
    # parser.add_argument("--experiment_name", type=str, help="Name of the experiment", default="ImageLingo")
    parser.add_argument("--model_type", type=str, help="Type of the model to register", default="decoder", choices=["encoder", "decoder"])
    parser.add_argument("--model_name", type=str, help="Name of the model to register", required=True)
    
    return parser.parse_args()

if __name__ == "__main__":
    
    args = parse_args()
    
    mlflow.set_tracking_uri(args.tracking_uri)
    # mlflow.set_experiment(args.experiment_name)
     
    register_model(args.run_id, args.best_artifact, args.model_type, args.model_name)
    
    print( f"Model {args.model_name} (type: {args.model_type}) registered successfully")
    
    