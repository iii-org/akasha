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
    import aiido
    if path_name is None:
        path_name = ""
    if "model" not in params or "embeddings" not in params:
        aiido.init(experiment=exp_name, run=path_name)

    else:
        mod = params["model"].split(":")
        emb = params["embeddings"].split(":")[0]
        sea = params["search_type"]
        aiido.init(experiment=exp_name,
                   run=emb + "-" + sea + "-" + "-".join(mod))

    aiido.log_params_and_metrics(params=params, metrics=metrics)

    if len(table) > 0:
        aiido.mlflow.log_table(table, "table.json")
    aiido.mlflow.end_run()
