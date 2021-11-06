## How to work with Command Line Arguments?
- If an optional argument is not passed, it's value will be extracted from configuration specified in the file `main.py` (based on `dataset_name`, `model_name`).
- If a valid argument value is passed through command line arguments, the code will use it further. That is, it will ignore the value assigned in the configuration.

## Command Line Arguments Information
| Argument name | Type | Valid Assignments | Default |
| --------------| ---- | ----------------- | ------- |
| dataset_name  | str  | azure, ett, etthourly, Solar, taxi30min, Traffic911 | positional argument|
| saved_models_dir       | str  | -                | None      |
| output_dir       | str  | -                | None      |
| N_input       | int  | >0                | -1      |
| N_output      | int  | >0                | -1      |
| epochs        | int  | >0                | -1      |
| normalize      | str  | same, zscore_per_series, gaussian_copula, log | None |
| learning_rate        | float  | >0                | -1.0      |
| hidden_size        | int  | >0                | -1      |
| num_grulstm_layers        | int  | >0                | -1      |
| batch_size        | int  | >0                | -1      |
| v_dim        | int  | >0                | -1      |
| t2v_type        | str  | local, idx, mdh_lincomb, mdh_parti | None      |
| K_list        | \[int,...,int \]  | \[>0,...,>0 \]               | \[\]      |
| device        | str  | -                | None      |

## Datasets
All the datasets can be found [here](https://drive.google.com/drive/folders/1b6xheczhJ1IwkTS5fqRf9_NkEkPf9beM?usp=sharing).

Add the dataset files/directories in `data` directory before running the code.

## Output files 
Following output files are stored in the `<output_dir>/<dataset_name>/` directory.

| File name | Description |
| --------- | ----------- |
| inputs.npy | Test input values, size: `number of time-series x N_input` |
| targets.npy | Test target/ground-truth values, size: `number of time-series x N_output` |
| `<model_name>`\_pred\_mu.npy | Mean forecast values. The size of the matrix is `number of time-series x number of time-steps` |
| `<model_name>`\_pred\_std.npy | Standard-deviation of forecast values. The size of the matrix is `number of time-series x number of time-steps` |


