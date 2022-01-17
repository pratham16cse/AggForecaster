#!/bin/bash

#python main.py taxi30min \
#	--N_input 336 --N_output 168 \
#	--saved_models_dir aaai_saved_models_taxi30min_e336_d168 \
#	--output_dir aaai_Outputs_taxi30min_e336_d168 \
#	--device cuda:0

#for K in 4 6 12; do
	#echo aaai_Outputs_ett_e384_d192_gpt_normmin_$K
#	--K_list 1 12 \
#python main.py ett \
#	--N_input 384 --N_output 192 \
#	--saved_models_dir aaai_saved_models_ett_e384_d192_gpt_normmin_sharqall \
#	--output_dir aaai_Outputs_ett_e384_d192_gpt_normmin_sharqall \
#	--K_list 1 2 3 4 6 8 12 24 \
#	--cv_inf 0 \
#	--device cuda:0
#done

python main.py ett \
	--N_input 384 --N_output 192 \
	--saved_models_dir ijcai_saved_models_ett \
	--output_dir ijcai_Outputs_ett \
	--cv_inf 0 \
	--device cuda:1

#for K in 2 6 12; do
	#echo aaai_Outputs_Solar_e336_d168_2_rp_$K
#	--K_list 1 6 \
#python main.py Solar \
#	--N_input 336 --N_output 168 \
#	--saved_models_dir aaai_saved_models_Solar_e336_d168_2_rp_sharqall \
#	--output_dir aaai_Outputs_Solar_e336_d168_2_rp_sharqall \
#	--K_list 1 2 3 4 6 8 12 24 \
#	--cv_inf 0 \
#	--device cuda:1
#done

python main.py Solar \
	--N_input 336 --N_output 168 \
	--saved_models_dir ijcai_saved_models_Solar \
	--output_dir ijcai_Outputs_Solar \
	--cv_inf 0 \
	--device cuda:1

#for K in 2 6; do
	#echo aaai_Outputs_etthourly_e168_d168_gpt_normmin_$K
#	--K_list 1 6 \
#python main.py etthourly \
#	--N_input 168 --N_output 168 \
#	--saved_models_dir aaai_saved_models_etthourly_e168_d168_gpt_normmin_sharqall \
#	--output_dir aaai_Outputs_etthourly_e168_d168_gpt_normmin_sharqall \
#	--K_list 1 2 3 4 6 8 12 24 \
#	--cv_inf 0 \
#	--device cuda:2
#done

python main.py etthourly \
	--N_input 168 --N_output 168 \
	--saved_models_dir ijcai_saved_models_etthourly \
	--output_dir ijcai_Outputs_etthourly \
	--cv_inf 0 \
	--device cuda:1

#for K in 2 6 12; do
#	#echo aaai_Outputs_electricity_e336_d168_testprune_2_rp_$K
#	--K_list 1 6 12 \
#python main.py electricity \
#	--N_input 336 --N_output 168 \
#	--saved_models_dir aaai_saved_models_electricity_e336_d168_testprune_2_rp_sharqall \
#	--output_dir aaai_Outputs_electricity_e336_d168_testprune_2_sharqall \
#	--K_list 1 2 3 4 6 8 12 24 \
#	--cv_inf 0 \
#	--device cuda:1
#done

python main.py electricity \
	--N_input 336 --N_output 168 \
	--saved_models_dir ijcai_saved_models_electricity \
	--output_dir ijcai_Outputs_electricity \
	--cv_inf 0 \
	--device cuda:1

#python main.py foodinflation \
#	--N_input 90 --N_output 30 \
#	--saved_models_dir saved_models_foodinflation_gpt_small_normzs_shiftmin \
#	--output_dir Outputs_foodinflation_gpt_small_normzs_shiftmin \
#	--K_list 1 \
#	--cv_inf 0 \
#	--device cuda:0

#python main.py foodinflationmonthly \
#	--N_input 12 --N_output 3 \
#	--saved_models_dir saved_models_foodinflationmonthly_gpt_small_normzs_shiftmin \
#	--output_dir Outputs_foodinflationmonthly_gpt_small_normzs_shiftmin \
#	--K_list 1 \
#	--cv_inf 0 \
#	--device cuda:0

# This dataset is used for testing/debugging
#python main.py aggtest \
#	--N_input 20 --N_output 20 \
#	--saved_models_dir aaai_saved_models_aggtest_e20_d20_gpt_conv_feats_nar \
#	--output_dir aaai_Outputs_aggtest_e20_d20_gpt_conv_feats_nar \
#	--K_list 1 \
#	--cv_inf 0 \
#	--device cuda:0

# Commands for Oracle and SimRetrieval
#python main.py ett \
#	--N_input 3840 --N_output 192 \
#	--saved_models_dir aaai_saved_models_ett_oracle \
#	--output_dir aaai_Outputs_ett_oracle \
#	--normalize same \
#	--device cuda:0

#python main.py Solar \
#	--N_input 1680 --N_output 168 \
#	--saved_models_dir aaai_saved_models_Solar_oracle \
#	--output_dir aaai_Outputs_Solar_oracle \
#	--normalize same \
#	--device cuda:0

#python main.py etthourly \
#	--N_input 840 --N_output 168 \
#	--saved_models_dir aaai_saved_models_etthourly_oracle \
#	--output_dir aaai_Outputs_etthourly_oracle \
#	--normalize same \
#	--device cuda:0
#
#python main.py electricity \
#	--N_input 1680 --N_output 168 \
#	--saved_models_dir aaai_saved_models_electricity_oracle \
#	--output_dir aaai_Outputs_electricity_oracle \
#	--normalize same \
#	--device cuda:0
