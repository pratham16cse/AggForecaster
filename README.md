## How to work with Command Line Arguments?
- If an argument is not passed, it's value will be extracted from configuration specified in the file `main.py` (based on `dataset_name`, `model_name`).
- If a valid argument value is passed through command line arguments, the code will use it further. That is, it will ignore the value assigned in the configuration.

## Command Line Arguments Information
| Argument name | Type | Valid Assignments | Default |
| --------------| ---- | ----------------- | ------- |
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

