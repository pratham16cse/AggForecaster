#!/bin/bash

#python main.py m4daily --epochs 20 --N_input 140 --N_output 14 --saved_models_dir saved_models_m4_normzscore_typeindustry_lr1e-3_numblocks8_period90 --output_dir Outputs_m4_normzscore_typeindustry_lr1e-3_numblocks8_period90 --normalize zscore_per_series --learning_rate 0.001 --batch_size 64 --hidden_size 128 --num_grulstm_layers 1 --v_dim 4 --device cuda:2
#python main.py m4daily --epochs 20 --N_input 140 --N_output 14 --saved_models_dir saved_models_m4_normzscore_typeindustry_lr1e-4_numblocks8_period90 --output_dir Outputs_m4_normzscore_typeindustry_lr1e-4_numblocks8_period90 --normalize zscore_per_series --learning_rate 0.0001 --batch_size 64 --hidden_size 128 --num_grulstm_layers 1 --v_dim 4 --device cuda:2
#python main.py m4daily --epochs 20 --N_input 140 --N_output 14 --saved_models_dir saved_models_m4_normzscore_typeindustry_lr1e-3_numblocks8_period90_coeffsstl --output_dir Outputs_m4_normzscore_typeindustry_lr1e-3_numblocks8_period90_coeffsstl --normalize zscore_per_series --learning_rate 0.001 --batch_size 64 --hidden_size 128 --num_grulstm_layers 1 --v_dim 4 --use_coeffs --device cuda:2
#python main.py m4daily --epochs 20 --N_input 140 --N_output 14 --saved_models_dir saved_models_m4_normzscore_typeindustry_lr1e-4_numblocks8_period90_coeffsstl --output_dir Outputs_m4_normzscore_typeindustry_lr1e-4_numblocks8_period90_coeffsstl --normalize zscore_per_series --learning_rate 0.0001 --batch_size 64 --hidden_size 128 --num_grulstm_layers 1 --v_dim 4 --use_coeffs --device cuda:2


#python main.py ett --epochs 20 --N_input 960 --N_output 96 --saved_models_dir saved_models_e960_d96_vdim8_K4_metricnll_normzscore --output_dir Outputs_e960_d96_vdim8_K4_metricnll_normzscore --normalize zscore_per_series --learning_rate 0.0001 --batch_size 64 --hidden_size 128 --num_grulstm_layers 1 --v_dim 8 --device cuda:2
#python main.py ett --epochs 20 --N_input 960 --N_output 96 --saved_models_dir saved_models_e960_d96_vdim8_K4_metricnll_normzscore_coeffsstl --output_dir Outputs_e960_d96_vdim8_K4_metricnll_normzscore_coeffsstl --normalize zscore_per_series --learning_rate 0.0001 --batch_size 64 --hidden_size 128 --num_grulstm_layers 1 --v_dim 8 --use_coeffs --device cuda:2
#python main.py ett --epochs 20 --N_input 960 --N_output 96 --saved_models_dir saved_models_e960_d96_vdim8_K4_metricnll_normzscore_lr1e-3 --output_dir Outputs_e960_d96_vdim8_K4_metricnll_normzscore_lr1e-3 --normalize zscore_per_series --learning_rate 0.001 --batch_size 64 --hidden_size 128 --num_grulstm_layers 1 --v_dim 8 --device cuda:2
#python main.py ett --epochs 20 --N_input 960 --N_output 96 --saved_models_dir saved_models_e960_d96_vdim8_K4_metricnll_normzscore_lr1e-3_coeffsstl --output_dir Outputs_e960_d96_vdim8_K4_metricnll_normzscore_lr1e-3_coeffsstl --normalize zscore_per_series --learning_rate 0.001 --batch_size 64 --hidden_size 128 --num_grulstm_layers 1 --v_dim 8 --use_coeffs --device cuda:2


#python main.py etthourly --epochs 20 --N_input 2400 --N_output 24 --saved_models_dir saved_models_etthourly --output_dir Outputs_etthourly --normalize zscore_per_series --learning_rate 0.0001 --batch_size 64 --hidden_size 128 --num_grulstm_layers 1 --v_dim 4 --device cuda:2
#python main.py etthourly --epochs 20 --N_input 2400 --N_output 24 --saved_models_dir saved_models_etthourly_coeffsstl --output_dir Outputs_etthourly_coeffsstl --normalize zscore_per_series --learning_rate 0.0001 --batch_size 64 --hidden_size 128 --num_grulstm_layers 1 --v_dim 4 --use_coeffs --device cuda:2
#python main.py etthourly --epochs 20 --N_input 2400 --N_output 24 --saved_models_dir saved_models_etthourly_coeffsstl_leak --output_dir Outputs_etthourly_coeffsstl_leak --normalize zscore_per_series --learning_rate 0.0001 --batch_size 64 --hidden_size 128 --num_grulstm_layers 1 --v_dim 4 --use_coeffs --device cuda:2
#python main.py etthourly --epochs 20 --N_input 2400 --N_output 24 --saved_models_dir saved_models_etthourly_coeffsstl_leak_unnorm --output_dir Outputs_etthourly_coeffsstl_leak_unnorm --normalize zscore_per_series --learning_rate 0.0001 --batch_size 64 --hidden_size 128 --num_grulstm_layers 1 --v_dim 4 --use_coeffs --device cuda:2


#python main.py ett --epochs 20 --N_input 960 --N_output 96 --saved_models_dir saved_models_ett --output_dir Outputs_ett --normalize zscore_per_series --learning_rate 0.0001 --batch_size 64 --hidden_size 128 --num_grulstm_layers 1 --v_dim 8 --device cuda:2
#python main.py ett --epochs 20 --N_input 960 --N_output 96 --saved_models_dir saved_models_ett_coeffsstl --output_dir Outputs_ett_coeffsstl --normalize zscore_per_series --learning_rate 0.0001 --batch_size 64 --hidden_size 128 --num_grulstm_layers 1 --v_dim 8 --use_coeffs --device cuda:2
#python main.py ett --epochs 20 --N_input 960 --N_output 96 --saved_models_dir saved_models_ett_coeffsstl_leak --output_dir Outputs_ett_coeffsstl_leak --normalize zscore_per_series --learning_rate 0.0001 --batch_size 64 --hidden_size 128 --num_grulstm_layers 1 --v_dim 8 --use_coeffs --device cuda:2
#python main.py ett --epochs 20 --N_input 960 --N_output 96 --saved_models_dir saved_models_ett_coeffsstl_leak_unnorm --output_dir Outputs_ett_coeffsstl_leak_unnorm --normalize zscore_per_series --learning_rate 0.0001 --batch_size 64 --hidden_size 128 --num_grulstm_layers 1 --v_dim 8 --use_coeffs --device cuda:2

#--t2v_type None
#python main.py taxi30min \
#	--N_input 336 --N_output 168 \
#	--saved_models_dir script_saved_models_taxi30min_e336_d168_t2vnone \
#	--output_dir script_Outputs_taxi30min_e336_d168_t2vnone \
#	--device cuda:2
#
#python main.py taxi30min \
#	--N_input 336 --N_output 168 \
#	--saved_models_dir script_saved_models_taxi30min_e336_d168_t2vlocal \
#	--output_dir script_Outputs_taxi30min_e336_d168_t2vlocal \
#	--t2v_type local \
#	--device cuda:2
#
#python main.py taxi30min \
#	--N_input 336 --N_output 168 \
#	--saved_models_dir script_saved_models_taxi30min_e336_d168_t2vglobalidx \
#	--output_dir script_Outputs_taxi30min_e336_d168_t2vglobalidx \
#	--t2v_type idx \
#	--device cuda:2
#
#python main.py taxi30min \
#	--N_input 336 --N_output 168 \
#	--saved_models_dir script_saved_models_taxi30min_e336_d168_t2vglobalmdhlincomb \
#	--output_dir script_Outputs_taxi30min_e336_d168_t2vglobalmdhlincomb \
#	--t2v_type mdh_lincomb \
#	--device cuda:2
#
#python main.py taxi30min \
#	--N_input 336 --N_output 168 \
#	--saved_models_dir script_saved_models_taxi30min_e336_d168_t2vglobalmdhparti \
#	--output_dir script_Outputs_taxi30min_e336_d168_t2vglobalmdhparti \
#	--t2v_type 'mdh_parti' \
#	--device cuda:2

#--t2v_type None
#python main.py ett \
#	--N_input 384 --N_output 192 \
#	--saved_models_dir saved_models_ett_e384_d192_t2vnone \
#	--output_dir Outputs_ett_e384_d192_t2vnone \
#	--device cuda:2
#
#python main.py ett \
#	--N_input 384 --N_output 192 \
#	--saved_models_dir saved_models_ett_e384_d192_t2vlocal \
#	--output_dir Outputs_ett_e384_d192_t2vlocal \
#	--t2v_type local \
#	--device cuda:2
#
#python main.py ett \
#	--N_input 384 --N_output 192 \
#	--saved_models_dir saved_models_ett_e384_d192_t2vglobalidx \
#	--output_dir Outputs_ett_e384_d192_t2vglobalidx \
#	--t2v_type idx \
#	--device cuda:2
#
#python main.py ett \
#	--N_input 384 --N_output 192 \
#	--saved_models_dir saved_models_ett_e384_d192_t2vglobalmdhlincomb \
#	--output_dir Outputs_ett_e384_d192_t2vglobalmdhlincomb \
#	--t2v_type mdh_lincomb \
#	--device cuda:2
#
#python main.py ett \
#	--N_input 384 --N_output 192 \
#	--saved_models_dir saved_models_ett_e384_d192_t2vglobalmdhparti \
#	--output_dir Outputs_ett_e384_d192_t2vglobalmdhparti \
#	--t2v_type mdh_parti \
#	--device cuda:2

#--t2v_type None
#python main.py azure \
#	--N_input 720 --N_output 360 \
#	--saved_models_dir saved_models_azure_e720_d360_t2vnone \
#	--output_dir script_Outputs_azure_e720_d360_t2vnone \
#	--device cuda:0

python main.py azure \
	--N_input 720 --N_output 360 \
	--saved_models_dir saved_models_azure_e720_d360_t2vlocal \
	--output_dir script_Outputs_azure_e720_d360_t2vlocal \
	--t2v_type local \
	--device cuda:0