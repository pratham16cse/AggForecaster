def add_metrics_to_dict(
	metrics_dict,
	model_name,
	metric_mse,
):
	if model_name not in metrics_dict:
		metrics_dict[model_name] = dict()

	metrics_dict[model_name]['mse'] = metric_mse

	return metrics_dict