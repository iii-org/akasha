from typing import Optional, Union, Dict, Sequence
import os
from pathlib import Path
from datetime import datetime
from importlib import metadata


def aiido_upload(
    exp_name,
    params: dict = {},
    metrics: dict = {},
    table: dict = {},
    path_name: str = "",
):
    """upload params_metrics, table to mlflow server for tracking.

    Args:
        **exp_name (str)**: experiment name on the tracking server, if not found, will create one .\n
        **params (dict, optional)**: parameters dictionary. Defaults to {}.\n
        **metrics (dict, optional)**: metrics dictionary. Defaults to {}.\n
        **table (dict, optional)**: table dictionary, used to compare text context between different runs in the experiment. Defaults to {}.\n
    """
    try:
        import mlflow

    except ImportError as e:
        print("mlflow is not installed. Please install it with 'pip install mlflow'.")
        raise e

    time_now = datetime.now().strftime("%Y%m%d%H%M%S")

    if path_name is None:
        path_name = ""
    if "model" not in params or "embeddings" not in params:
        mlflow_init(experiment=exp_name, run=path_name + "-" + time_now)

    else:
        mod = params["model"].split(":")
        emb = params["embeddings"].split(":")[0]
        sea = params["search_type"]
        mlflow_init(
            experiment=exp_name,
            run=emb + "-" + sea + "-" + "-".join(mod) + "-" + time_now,
        )

    log_params_and_metrics(params=params, metrics=metrics)

    if len(table) > 0:
        mlflow.log_table(table, "table.json")
    mlflow.end_run()


def mlflow_init(
    experiment: Optional[str] = None,
    run: Optional[str] = None,
    load_config_from_dotenv: Union[bool, str] = False,
    do_not_raise: bool = True,
):
    try:
        import mlflow

        try:
            __version__ = metadata.version("akasha-light")
        except Exception:
            __version__ = "dev"

    except ImportError:
        print("package is not installed. Please install it with 'pip install mlflow'.")
        if not do_not_raise:
            raise
        return

    try:
        if load_config_from_dotenv:
            from dotenv import load_dotenv

            if isinstance(load_config_from_dotenv, str):
                res = load_dotenv(load_config_from_dotenv, override=True)
            elif load_config_from_dotenv is True:
                res = load_dotenv(os.path.join(os.getcwd(), ".env"), override=True)
            else:
                res = False
            if not res:
                raise RuntimeError("No environment variables were loaded.")
    except Exception as e:
        print(
            f"Error loading environment variables from {load_config_from_dotenv if isinstance(load_config_from_dotenv, str) else '.env'}\n{e}"
        )

    try:
        time_now = datetime.now().strftime("%Y%m%d%H%M%S")
        mlflow.set_tracking_uri(
            os.getenv("TRACKING_SERVER_URI", Path(os.path.join(os.getcwd(), "mlruns")))
        )
        tracking_uri = mlflow.get_tracking_uri()
        if tracking_uri.endswith("mlruns") and tracking_uri.startswith("file:"):
            print(
                f"Remote TRACKING_SERVER_URI is not set, use default local value: {tracking_uri}"
            )

        if experiment is not None:
            mlflow.set_experiment(experiment)

        if run is None or run == "":
            run = "mlflow-" + time_now
        mlflow.set_tag("mlflow.runName", run)
        mlflow.set_tag("_Software.akasha.SDK", __version__)
        print(
            f"\n[{time_now}][OK][init] mlflow initialized with experiment: {experiment}, run: {run}"
        )
    except Exception as e:
        print(
            f"[{time_now}][Fail][init] Failed initializing mlflow with experiment: {experiment}, run: {run}"
        )
        if not do_not_raise:
            raise e


def log_params_and_metrics(params: Dict, metrics: Union[Dict, Sequence[Dict]]):
    """Equivalent to mlflow.log_params then mlflow.log_metrics

    Parameters
    ----------
    params : Dict
        Training parameters, such as learning rate, weight decay, etc.
    metrics : Dict
        Evaluation metrics, such as accuracy, top-5 accuracy, etc. or a list of such metrics
    """
    # Check if mlflow is installed
    import mlflow

    try:
        time_now = datetime.now().strftime("%Y%m%d%H%M%S")
        mlflow.log_params(params)

        try:
            if isinstance(metrics, Sequence):
                for metric in metrics:
                    mlflow.log_metrics(metric)
            else:
                mlflow.log_metrics(metrics)
            print(f"[{time_now}][OK][log_metrics] Metrics logged")
        except Exception as e:
            print(f"[{time_now}][Fail][log_metrics] {e}")
            raise e

        print(f"[{time_now}][OK][log_params_and_metrics] Params and metrics logged\n")

    except Exception as e:
        print(
            f"[{time_now}][Fail][log_params_and_metrics] Failed logging params and metrics: {e}\n"
        )
        raise e
