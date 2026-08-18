import argparse
import yaml
from pathlib import Path
from dataclasses import asdict

from utils.ExpConfigs import ExpConfigs

'''
variable `configs` are ONLY determined by argparse, not .yaml files.

.yaml files are automatically saved and maintained as references:

- configs.yaml in training directory:

    Exact configs used in that training.

- *.yaml in "configs" folder

    Reference default configs for each model and dataset combination. Only automatically updated if new argument are added in argparse.
'''

parser = argparse.ArgumentParser(description='Irregular Time Series Forecasting')

# basic config
parser.add_argument('--task_name', type=str, choices=["long_term_forecast", "short_term_forecast", "imputation", "classification", "anomaly_detection", "representation_learning"], default='short_term_forecast', help='task name')
parser.add_argument('--is_training', type=int, default=1, help='training or testing')
parser.add_argument('--model_id', type=str, default='LSTM', help='model id')
parser.add_argument('--model_name', type=str, default='LSTM', help='model name')
parser.add_argument('--checkpoints', type=str, default='storage/results/', help='where to save model checkpoints in training')
parser.add_argument('--ablation_name', type=str, default='', help='name for the ablation study, used for creating a separate directory')
# dataset & data loader
parser.add_argument('--dataset_name', type=str, default='ETTm1', help='dataset type')
parser.add_argument('--dataset_root_path', type=str, default='storage/datasets/ETT/', help='root path of the data file')
parser.add_argument('--dataset_file_name', type=str, default=None, help='data file name')
parser.add_argument('--features', type=str, choices=['M', 'S', "MS"], default='M', help='forecasting task; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target_variable_name', type=str, default="OT", help='target variable name in regular time series datasets. Originally named as --target.')
parser.add_argument('--target_variable_index', type=int, default=0, help='target variable index in datasets. Should not be used together with target_variable_name')
parser.add_argument('--freq', type=str, choices=['s', 't', 'h', 'd', 'b', 'w', 'm'], default='h', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--collate_fn', type=str, default="collate_fn", help='Name of the function as a custom collate_fn for dataloader. By default, datasets without collate_fn defined in data/data_provider/datasets/SOME_DATASET_NAME will use default collate_fn of Pytorch. Refer to data/data_provider/data_factory.py for implementation detail.')
parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
parser.add_argument('--missing_rate', type=float, default=0., help="Manually mask out some observations.")
parser.add_argument('--train_val_loader_shuffle', type=int, default=None, help="By default, this parameter is unset, and train and val loader is shuffled.")
parser.add_argument('--train_val_loader_drop_last', type=int, default=None, help="By default, this parameter is unset, and train and val loader will drop the last batch if the number of samples is not sufficient.")
parser.add_argument('--pin_memory', type=int, default=1, help='pin CPU dataloader memory before CUDA transfer')
parser.add_argument('--persistent_workers', type=int, default=1, help='keep dataloader workers alive between epochs when num_workers > 0')
parser.add_argument('--prefetch_factor', type=int, default=4, help='number of batches prefetched by each dataloader worker')
parser.add_argument('--non_blocking_transfer', type=int, default=1, help='use non-blocking host-to-GPU tensor copies when pin_memory is enabled')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=0, help='start token length. Should be zero for tasks other than forecasting')
parser.add_argument('--pred_len', type=int, default=0, help='prediction sequence length. Should be zero for tasks other than forecasting')

# classification task
parser.add_argument('--n_classes', type=int, default=2, help='number of classes')

# GPU
parser.add_argument('--use_gpu', type=int, default=1, help='use gpu')
parser.add_argument('--gpu_id', type=int, default=0, help='primary gpu id, will be overwritten when use_multi_gpu is 1. Originally named as --gpu.')
parser.add_argument('--use_multi_gpu', type=int, help='use multiple gpus, via huggingface accelerate library', default=0)
parser.add_argument('--gpu_ids', type=str, default=None, help='string of device ids for multile gpus. Originally named as --devices.')
parser.add_argument('--tf32', type=int, default=1, help='enable TF32 matmul/cuDNN kernels on Ampere or newer GPUs')
parser.add_argument('--cudnn_benchmark', type=int, default=1, help='enable cuDNN benchmark for faster fixed-shape kernels')
parser.add_argument('--amp_dtype', type=str, default='off', choices=['off', 'bf16'], help='optional CUDA autocast dtype for faster experiments')

# training
parser.add_argument('--wandb', type=int, default=0, help='whether to use weight & bias for monitoring')
parser.add_argument('--sweep', type=int, default=0, help='whether to use weight & bias for hyperparameter searching')
parser.add_argument('--val_interval', type=int, default=1, help='validation interval relative to training epochs')
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--disable_tqdm', type=int, default=0, help='disable tqdm progress bars to reduce host overhead in many-process runs')
parser.add_argument('--seed_base', type=int, default=2024, help='base random seed; itr i uses seed_base + i')
parser.add_argument('--itr', type=int, default=5, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=300, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='initial learning rate')
parser.add_argument('--loss', type=str, default='MSE', help='loss function, should be exactly the same as the file name under loss_fns directory')
parser.add_argument('--mae_weight', type=float, default=0.20,
                    help='Weight of the MAE term for joint MSEMAE loss.')
parser.add_argument('--lr_scheduler', type=str, choices=["ExponentialDecayLR", "ManualMilestonesLR", "DelayedStepDecayLR", "CosineAnnealingLR", "MultiStepLR", "constant"], default='DelayedStepDecayLR', help='learning rate scheduler. Originally named as --lradj')
parser.add_argument('--pretrained_checkpoint_root_path', type=str, default="", help="Path to folder containing pretrained model's checkpoints")
parser.add_argument('--pretrained_checkpoint_file_name', type=str, default="", help="file name of pretrained model's checkpoints, including file type extension")
parser.add_argument('--n_train_stages', type=int, default=1, help="Some models have multiple training stages, like pretraining + finetuning. e.g., --n_train_stages 2 will pass train_stage=1 and train_stage=2 to model during training.")
parser.add_argument('--retain_graph', type=int, default=0, help='whether to retain compute graph in back propagation. Used in special models like HD_TTS.')

# testing
parser.add_argument('--checkpoints_test', type=str, default=None, help='folder where model checkpoint file is saved, for testing')
parser.add_argument('--test_all', type=int, default=0, help='whether to test on all train, val, and test sets')
parser.add_argument('--test_split', type=str, default='test', choices=['test', 'val', 'train', 'test_all'],
                    help='dataset split used by Exp_Main.test; test_all keeps legacy behavior when --test_all=1')
parser.add_argument('--test_flop', type=int, default=0, help='Test model flops. See utils/tools for usage')
parser.add_argument('--test_train_time', type=int, default=0, help="Test model's training time. See utils/tools for usage")
parser.add_argument('--test_gpu_memory', type=int, default=0, help="Test model's gpu memory usage. See utils/tools for usage")
parser.add_argument('--test_dataset_statistics', type=int, default=0, help="Test dataset's statistics.")
parser.add_argument('--test_zero_shot', type=int, default=0, help='Test zero-shot performance. i.e., without training.')
parser.add_argument('--save_arrays', type=int, default=0, help='whether to save model input and output as .npy files, for later visualization')
parser.add_argument('--load_checkpoints_test', type=int, default=1, help='whether to load checkpoint during testing')
parser.add_argument('--test_inference_time', type=int, default=0, help="Test model's inference time. See utils/tools for usage")
# model configs
# common
parser.add_argument('--patch_len', type=int, default=12, help='patch length. Also used as period_len in some models (SparseTSF).')
parser.add_argument('--patch_stride', type=int, default=12, help='stride when splitting patches. Originally named as --stride.')
parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
parser.add_argument('--revin_affine', type=int, default=0, help='RevIN-affine; True 1 False 0. Originally named as --affine.')
parser.add_argument('--kernel_size', type=int, default=25, help='kernel size')
parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')
parser.add_argument('--channel_independence', type=int, default=1, help='0: channel dependence 1: channel independence for FreTS/TimeMixer model')
parser.add_argument('--scale_factor', type=int, default=2, help='scale factor for upsample')
parser.add_argument('--top_k', type=int, default=5, help='top k selection')
parser.add_argument('--embed_type', type=int, choices=[0, 1, 2, 3, 4], default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
parser.add_argument('--enc_in', type=int, default=2, help='encoder input size / input time series number of variables. In most cases, it should be adjusted per dataset') 
parser.add_argument('--dec_in', type=int, default=2, help='decoder input size. Usually it is the same as --enc_in')
parser.add_argument('--c_out', type=int, default=2, help='output size. Usually it is the same as --enc_in')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--d_timesteps', type=int, default=1, help='UNUSED. Size of last dimension of `x_mark`/`y_mark`. Many Regular/Spatiotemporal datasets stack time in day, day in week, etc. along the last dimension. Others default to size 1.')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads (in attention)')
parser.add_argument('--n_layers', type=int, default=1, help='num of layers')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--hidden_layers', type=int, default=1, help='Number of hidden layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--embed', type=str, choices=["timeF", "fixed", "learned"], default='timeF', help='time features encoding')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', type=int, default=0, help='output attention weight')
parser.add_argument('--node_dim', type=int, default=10, help='hidden dimension of nodes used in a few GNNs, like tPatchGNN')
# PatchTST
parser.add_argument('--patchtst_fc_dropout', type=float, default=0.05, help='fully connected dropout')
parser.add_argument('--patchtst_head_dropout', type=float, default=0.0, help='head dropout')
parser.add_argument('--patchtst_padding_patch', default='end', help='None: None; end: padding on the end')
parser.add_argument('--patchtst_subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
parser.add_argument('--patchtst_decomposition', type=int, default=0, help='decomposition; True 1 False 0')
# Mamba
parser.add_argument('--mamba_d_conv', type=int, default=4, help='conv kernel size for Mamba')
parser.add_argument('--mamba_expand', type=int, default=2, help='expansion factor for Mamba')
# Latent ODE
parser.add_argument('--latent_ode_units', type=int, default=100, help="Number of units per layer in ODE func")
parser.add_argument('--latent_ode_gen_layers', type=int, default=1, help="Number of layers in ODE func in generative ODE")
parser.add_argument('--latent_ode_rec_layers', type=int, default=1, help="Number of layers in ODE func in recognition ODE")
parser.add_argument('--latent_ode_z0_encoder', type=str, default='odernn', help="Type of encoder for Latent ODE model: odernn or rnn")
parser.add_argument('--latent_ode_rec_dims', type=int, default=20, help="Dimensionality of the recognition model (ODE or RNN).")
parser.add_argument('--latent_ode_gru_units', type=int, default=100, help="Number of units per layer in each of GRU update networks")
parser.add_argument('--latent_ode_classif', type=int, default=0, help="Include binary classification loss -- used for Physionet dataset for hospiral mortality")
parser.add_argument('--latent_ode_linear_classif', type=int, default=0, help="If using a classifier, use a linear classifier instead of 1-layer NN")
# CRU
parser.add_argument('--cru_num_basis', type=int, default=15, help="Number of basis matrices to use in transition model for locally-linear transitions. K in paper")
parser.add_argument('--cru_bandwidth', type=int, default=3, help="Bandwidth for basis matrices A_k. b in paper")
parser.add_argument('--cru_ts', type=float, default=1.0, help="Scaling factor of timestamps for numerical stability.")
# NeuralFlows
parser.add_argument('--neuralflows_flow_model', type=str, default='coupling', help='Type of NeuralFlows model', choices=['coupling', 'resnet', 'gru'])
parser.add_argument('--neuralflows_flow_layers', type=int, default=1, help='Number of flow layers')
parser.add_argument('--neuralflows_latents', type=int, default=20, help='Size of the latent state')
parser.add_argument('--neuralflows_time_net', type=str, default='TimeLinear', help='Name of time net', choices=['TimeFourier', 'TimeFourierBounded', 'TimeLinear', 'TimeTanh'])
parser.add_argument('--neuralflows_time_hidden_dim', type=int, default=1, help='Number of time features (only for Fourier)')
# PrimeNet
parser.add_argument('--primenet_pooling', type=str, default='ave', help='[ave, att, bert]: What pooling to use to aggregate the model output sequence representation for different tasks.')
# mTAN
parser.add_argument('--mtan_num_ref_points', type=int, default=8, help='number of reference points, originally chosen in [8, 16, 32, 64, 128]')
parser.add_argument('--mtan_alpha', type=float, default=100., help='In classification task, loss is calculated as recon_loss + self.alpha * ce_loss')
# TimeMixer
parser.add_argument('--timemixer_decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
parser.add_argument('--timemixer_use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
parser.add_argument('--timemixer_down_sampling_layers', type=int, default=0, help='num of down sampling layers')
parser.add_argument('--timemixer_down_sampling_method', type=str, default="avg",
                        help='down sampling method, only support avg, max, conv')
# Nonstationary Transformer
parser.add_argument('--nonstationarytransformer_p_hidden_dims', type=int, nargs='+', default=[128, 128],
                    help='hidden layer dimensions of projector (List)')
parser.add_argument('--nonstationarytransformer_p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')
parser.add_argument('--informer_distil', type=int,
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=1)
# tPatchGNN
parser.add_argument('--tpatchgnn_te_dim', type=int, default=10, help="Number of units for time encoding")
# SPECTRON (Patched Version)
parser.add_argument('--spectron_num_kernels', type=int, default=64, help='K: Number of spectral basis functions per patch.')
parser.add_argument('--spectron_d_max', type=float, default=5.0, help='Maximum absolute value for the decay/growth factor d in basis functions.')
parser.add_argument('--spectron_patch_len', type=int, default=64, help='P: Length of each patch (number of points).')
parser.add_argument('--spectron_patch_stride', type=int, default=32, help='S: Stride between consecutive patches.')
parser.add_argument('--spectron_num_intra_layers', type=int, default=2, help='Number of layers in the Patch-Spectral Transformer for processing the sequence of patch spectra.')# Used to be compatible with ipython. Never used
parser.add_argument('--spectron_kernel_chunk_size', type=int, default=16, help='Chunk size for processing kernels to save memory.')
parser.add_argument('--spectron_num_last_patches', type=int, default=3, help='N: Number of last patches to use for robust prediction.')
# TAC-Mixer
parser.add_argument('--tac_patch_num', type=int, default=50,
                    help='P in the paper. Number of time patches the timeline is divided into.')
parser.add_argument('--tac_mixer_hidden_dim_p', type=int, default=64,
                    help='D_P in the paper. Hidden dimension of the token-mixing MLP in the Temporal Mixer.')
parser.add_argument('--tac_mixer_hidden_dim_c', type=int, default=32,
                    help='D_C in the paper. Hidden dimension of the variable-mixing MLP in the Variable Mixer.')
parser.add_argument('--tac_decoder_context_k', type=int, default=1,
                    help='k in the paper. The number of neighboring patches to consider on each side during decoding (k=1 means a window of 3 patches).')
# ASTGI
parser.add_argument('--astgi_k_neighbors', type=int, default=64,
                    help='Number of K-nearest neighbors for graph construction in ST-PPGN.')
parser.add_argument('--astgi_prop_layers', type=int, default=2,
                    help='Number of spatio-temporal propagation layers in ST-PPGN.')
parser.add_argument('--astgi_channel_dim', type=int, default=64,
                    help='Dimension of the channel embedding (d_c) in ST-PPGN.')
parser.add_argument('--astgi_time_dim', type=int, default=64,
                    help='Dimension of the time encoding (d_t) in ST-PPGN.')
parser.add_argument('--astgi_mlp_ratio', type=float, default=2.0,
                    help='Ratio of the hidden dimension to the model dimension in all MLPs for ST-PPGN.')
parser.add_argument('--astgi_channel_dist_weight', type=float, default=1.0,
                    help='Weight (w_c) for channel distance in K-NN search. Time distance weight (w_t) is fixed to 1.')
# APN
parser.add_argument('--apn_te_dim', type=int, default=32,
                    help='Dimension of the learnable time embedding (te_dim) in APN.')
parser.add_argument('--apn_npatch', type=int, default=16,
                    help='Number of patches (P) to create from the time series in APN.')
parser.add_argument('--apn_patch_size', type=float, default=16.0,
                    help='Base size (S) of each patch in APN. This determines the initial temporal width.')
parser.add_argument('--apn_nlayer', type=int, default=2,
                    help='Number of layers in the core APN model.')
parser.add_argument('--apn_attn_heads', type=int, default=8,
                    help='Number of attention heads for cross-variable aggregation in APN.')
parser.add_argument('--apn_research_variant', type=str, default='base',
                    help='APNResearch variant. Use "__" to compose lightweight modules, e.g. adaptive_fourier__patchconv__lowrank.')
parser.add_argument('--apn_mcr_gate_init', type=float, default=-1.6,
                    help='Initial gate logit for APNResearch mechanism-conditioned residual heads.')
parser.add_argument('--apn_mcr_residual_l2', type=float, default=0.0,
                    help='Optional L2 regularization weight on mechanism-conditioned residual outputs.')
parser.add_argument('--apn_mcr_integral_centers', type=int, default=6,
                    help='Number of RBF centers for the mechanism-conditioned integral residual head.')
parser.add_argument('--apn_mcr_integral_width', type=float, default=0.18,
                    help='RBF width for the mechanism-conditioned integral residual head.')
parser.add_argument('--apn_alias_remainder_scale', type=float, default=0.25,
                    help='Bound on the tanh remainder branch in alias-factored LARM residual heads.')
parser.add_argument('--apn_alias_remainder_l2', type=float, default=0.0,
                    help='Optional L2 penalty applied only to the alias-factored LARM remainder branch.')
parser.add_argument('--apn_alias_score_gate_weight', type=float, default=1.0,
                    help='Weight on log(1+alias_score) in the alias-factored LARM gate.')
parser.add_argument('--apn_alias_variance_price_weight', type=float, default=0.0,
                    help='Weight for label-free variance pricing in alias-factored LARM gates; subtracts log residual energy normalized by level scale.')
parser.add_argument('--apn_alias_scale_price_weight', type=float, default=0.0,
                    help='Optional weight for penalizing high level-scale contexts in alias-factored LARM gates.')
parser.add_argument('--apn_lamr_motion_scale', type=float, default=1.0,
                    help='Motion-prior scale for APNResearch LAMR residual head.')
parser.add_argument('--apn_lamr_correction_scale', type=float, default=0.10,
                    help='Bounded neural correction scale for APNResearch LAMR residual head.')
parser.add_argument('--apn_diff_loss_weight', type=float, default=0.0,
                    help='Auxiliary masked first-difference trajectory loss weight for APNResearch forecasts.')
parser.add_argument('--apn_diff2_loss_weight', type=float, default=0.0,
                    help='Auxiliary masked second-difference trajectory loss weight for APNResearch forecasts.')
parser.add_argument('--apn_diff_loss_beta', type=float, default=0.05,
                    help='Smooth-L1 beta for APNResearch trajectory-difference auxiliary losses.')
parser.add_argument('--apn_kinematic_gate_init', type=float, default=-2.0,
                    help='Initial gate logit for APNResearch kinematic decoder blending.')
parser.add_argument('--apn_kinematic_slope_scale', type=float, default=0.25,
                    help='Scale of learned velocity correction in APNResearch kinematic decoder.')
parser.add_argument('--apn_kinematic_accel_scale', type=float, default=0.05,
                    help='Scale of learned curvature term in APNResearch kinematic decoder.')
parser.add_argument('--apn_kinematic_slope_clip', type=float, default=10.0,
                    help='Absolute clip for history-estimated velocity in APNResearch kinematic decoder.')
parser.add_argument('--apn_mcr_state_mode', type=str, default='state',
                    choices=['state', 'prediction_only'],
                    help='Residual state interface: state uses backbone state; prediction_only zeros backbone state so residual sees only final prediction, time, and mechanism summaries.')
parser.add_argument('--apn_mcg_gate_init', type=float, default=-2.0,
                    help='Initial integral-branch gate logit for adaptive MCR/MCIR residual mixing.')
parser.add_argument('--apn_mcg_warmup_stage', type=int, default=0,
                    help='Use a fixed MCR/MCIR blend through this train stage before learning the adaptive gate.')
parser.add_argument('--apn_mcg_warmup_blend', type=float, default=0.5,
                    help='Fixed MCIR branch weight used during adaptive-gate warmup stages.')
parser.add_argument('--apn_mcg_boundary_weight', type=float, default=0.0,
                    help='Auxiliary weight for g*(1-g) boundary regularization on adaptive MCR/MCIR gate.')
parser.add_argument('--apn_mcg_supervised_weight', type=float, default=0.0,
                    help='Auxiliary weighted BCE loss for supervising adaptive gate with per-batch branch-quality targets.')
parser.add_argument('--apn_mcg_supervised_temperature', type=float, default=0.10,
                    help='Relative-loss temperature for soft MCR-vs-MCIR branch-quality gate targets.')
parser.add_argument('--apn_mcg_supervised_margin', type=float, default=0.0,
                    help='Optional relative-loss margin before the MCIR branch is treated as better for supervised gate targets.')
parser.add_argument('--apn_mcg_supervised_min_confidence', type=float, default=0.0,
                    help='Drop supervised gate targets whose soft-label confidence is below this threshold.')
parser.add_argument('--apn_query_count', type=int, default=4,
                    help='Number of per-variable queries used by APNResearch querymix.')
parser.add_argument('--apn_varmix_init', type=float, default=-3.0,
                    help='Initial residual gate logit for APNResearch variable mixing.')
parser.add_argument('--apn_fourier_init', type=float, default=-2.0,
                    help='Initial residual gate logit for APNResearch Fourier modulation.')
parser.add_argument('--apn_reliability_source_scale', type=float, default=1.0,
                    help='Scale for source-variable reliability bias in APNResearch reliability-aware variable mixing.')
parser.add_argument('--apn_reliability_target_scale', type=float, default=1.0,
                    help='Scale for target-variable uncertainty gate in APNResearch reliability-aware variable mixing.')
parser.add_argument('--apn_reliability_impute_weight', type=float, default=0.15,
                    help='Soft weight for reliability-conditioned imputed observations in APNResearch patch aggregation.')
parser.add_argument('--apn_reliability_route_weight', type=float, default=0.0,
                    help='Penalty weight for routing information through low-reliability source variables.')
parser.add_argument('--apn_selector_hidden', type=int, default=32,
                    help='Hidden dimension of the SAPN structural selector.')
parser.add_argument('--apn_selector_l1', type=float, default=1e-4,
                    help='L1 gate regularization for the SAPN structural selector.')
parser.add_argument('--apn_selector_residual', type=float, default=1e-4,
                    help='Residual magnitude regularization for SAPN selected branches.')
parser.add_argument('--apn_selector_entropy', type=float, default=0.0,
                    help='Optional entropy penalty for SAPN selector gates.')
parser.add_argument('--apn_selector_level', type=str, default='variable', choices=['sample', 'variable'],
                    help='Apply SAPN branch gates per sample or per sample-variable pair.')
parser.add_argument('--apn_selector_init', type=float, default=-2.0,
                    help='Initial gate logit bias for the SAPN structural selector.')
parser.add_argument('--apn_selector_scale', type=float, default=1.0,
                    help='Global multiplier for SAPN selector gates; values below 1 preserve a strong prior backbone.')
parser.add_argument('--apn_selector_safety', type=int, default=0,
                    help='Whether to learn an extra no-op safety gate that can suppress all SAPN selector residuals.')
parser.add_argument('--apn_selector_safety_init', type=float, default=-3.0,
                    help='Initial logit bias for the SAPN no-op safety gate.')
parser.add_argument('--apn_selector_temperature', type=float, default=1.0,
                    help='Temperature for SAPN branch selector sigmoid gates.')
parser.add_argument('--apn_selector_safety_temperature', type=float, default=1.0,
                    help='Temperature for SAPN safety gate sigmoid.')
parser.add_argument('--apn_selector_branch_mask', type=str, default='all',
                    help='Comma-separated SAPN branches to enable: spectral,trend,cov,decoder; use all for every branch.')
parser.add_argument('--apn_selector_branch_dropout', type=float, default=0.0,
                    help='Training-time dropout probability applied to effective SAPN branch gates.')
parser.add_argument('--apn_selector_stat_control', type=str, default='real',
                    help='Structural-statistic control for SAPN: real, shuffle, random, density_only, uncertainty_only, or global.')
parser.add_argument('--apn_selector_uncertainty_weight', type=float, default=0.0,
                    help='Strength of uncertainty-based safety shrinkage for SAPN gates.')
parser.add_argument('--apn_selector_uncertainty_mode', type=str, default='none',
                    help='Uncertainty score for safety shrinkage: none, robust, interp, quad, smooth, or all.')
parser.add_argument('--apn_selector_trust_weight', type=float, default=0.0,
                    help='Hinge trust-region penalty weight for residual energy above --apn_selector_trust_cap.')
parser.add_argument('--apn_selector_trust_cap', type=float, default=0.0,
                    help='Residual energy cap for the SAPN trust-region penalty.')
parser.add_argument('--apn_selector_mass_weight', type=float, default=0.0,
                    help='Weight for keeping the effective SAPN open mass inside a target band during candidate training.')
parser.add_argument('--apn_selector_mass_min', type=float, default=0.0,
                    help='Lower target for average effective SAPN open mass when --apn_selector_mass_weight is positive.')
parser.add_argument('--apn_selector_mass_max', type=float, default=1.0,
                    help='Upper target for average effective SAPN open mass when --apn_selector_mass_weight is positive.')
parser.add_argument('--apn_operator_price_weight', type=float, default=1.0,
                    help='Weight of the explicit risk price in SAPN operator-correction gates.')
parser.add_argument('--apn_operator_price_margin', type=float, default=0.0,
                    help='Extra safe-benefit margin subtracted before opening SAPN operator-correction gates.')
parser.add_argument('--apn_operator_price_init', type=float, default=0.1,
                    help='Initial positive coefficient scale for SAPN operator-correction price components.')
parser.add_argument('--apn_operator_dep_branch', type=str, default='cov',
                    help='Dependency correction branch for sapn_operator: cov, graph, or hybrid.')
parser.add_argument('--apn_emit_noop_pred', type=int, default=0,
                    help='During APNResearch test, also emit the same-checkpoint hard-noop prediction obtained by zeroing selector residuals.')
parser.add_argument('--apn_emit_selector_arrays', type=int, default=0,
                    help='During APNResearch test, emit per-sample selector gates for paired certificate auditing.')
parser.add_argument('--apn_emit_mcg_arrays', type=int, default=0,
                    help='During APNResearch test, emit adaptive MCR/MCIR gate and branch predictions for diagnostics.')
parser.add_argument('--apn_paired_safe_weight', type=float, default=0.0,
                    help='Weight for paired no-negative-transfer penalty in paired-safe losses.')
parser.add_argument('--apn_paired_safe_margin', type=float, default=0.0,
                    help='Margin for paired no-negative-transfer penalty in paired-safe losses.')
parser.add_argument('--apn_gate_harm_weight', type=float, default=0.0,
                    help='Weight for penalizing selector gate mass on samples where the paired candidate currently harms the backbone.')
parser.add_argument('--apn_gate_bce_weight', type=float, default=0.0,
                    help='Weight for online detached benefit-label BCE supervision of selector gate scores.')
parser.add_argument('--apn_gate_target_margin', type=float, default=0.0,
                    help='Detached per-sample benefit margin used to form online selector targets.')
parser.add_argument('--apn_paired_cvar_q', type=float, default=1.0,
                    help='Upper-tail fraction of positive paired excess used by CVaR-style paired-safe losses.')
parser.add_argument('--apn_paired_benefit_weight', type=float, default=0.0,
                    help='Weight for smooth paired-benefit reward in balanced paired-value losses.')
parser.add_argument('--apn_paired_benefit_margin', type=float, default=0.0,
                    help='Required paired benefit before the smooth benefit reward becomes strong.')
parser.add_argument('--apn_paired_benefit_temperature', type=float, default=1e-3,
                    help='Softplus temperature for smooth paired-benefit reward.')
parser.add_argument('--apn_paired_benefit_cap', type=float, default=0.005,
                    help='Per-sample cap for the smooth paired-benefit reward.')
parser.add_argument('--apn_paired_lcb_var_weight', type=float, default=0.0,
                    help='Weight for empirical-Bernstein-style paired-benefit variance control.')
parser.add_argument('--apn_paired_lcb_var_cap', type=float, default=0.01,
                    help='Symmetric clipping radius used before paired-benefit variance control.')
parser.add_argument('--apn_paired_ratio_rho', type=float, default=0.2,
                    help='Batch quantile used as the floor for studentized paired losses.')
parser.add_argument('--apn_paired_ratio_eps', type=float, default=1e-4,
                    help='Numerical lower floor for studentized paired-loss denominators.')
parser.add_argument('--apn_paired_ratio_cap', type=float, default=0.05,
                    help='Per-sample cap for smooth reward in studentized paired-loss units.')
parser.add_argument('--apn_paired_ntr_weight', type=float, default=0.0,
                    help='Weight for smooth no-negative-transfer frequency control in paired losses.')
parser.add_argument('--apn_paired_ntr_temperature', type=float, default=1e-3,
                    help='Sigmoid temperature for smooth no-negative-transfer frequency control.')
# MCOR black-box wrapper
parser.add_argument('--mcor_base_model_name', type=str, default='KAFNet',
                    help='Base forecasting model wrapped by MCORWrapper for black-box mechanism-conditioned residuals.')
parser.add_argument('--mcor_force_branch', type=str, default='none', choices=['none', 'local', 'integral', 'mix'],
                    help='Force MCORWrapper branch routing for ablation: none learns the gate, local/integral force one branch, mix uses the warmup blend.')
parser.add_argument('--mcor_freeze_base', type=int, default=0,
                    help='Freeze the wrapped base model and load plain base checkpoints into MCORWrapper as base_model.* weights.')
parser.add_argument('--mcor_reliability_mode', type=str, default='real',
                    choices=['real', 'value_only', 'reliability_only', 'shuffled', 'random', 'constant'],
                    help='Control the RAMS mechanism state for theory ablations: real, value_only, reliability_only, shuffled, random, or constant.')
configs = ExpConfigs(**vars(parser.parse_args())) # enable type hints

# .yaml reference file maintainance
yaml_configs_path_deprecated = Path(f"configs/{configs.model_name}/{configs.dataset_name}.yaml") # backward compatibility
yaml_configs_path = Path(f"configs/{configs.model_name}/{configs.model_id}/{configs.dataset_name}.yaml")
if yaml_configs_path.exists():
    with open(yaml_configs_path, 'r', encoding="utf-8") as stream:
        try:
            yaml_configs: dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(f"utils/configs.py: Exception when parsing {yaml_configs_path}: {exc}")
            exit(1)
    if yaml_configs is not None:
        # update yaml only if new args are added in argparse
        if_update = False
        for key, value in configs.__dict__.items():
            if key not in yaml_configs.keys():
                if_update = True
                yaml_configs[key] = value

        if if_update:
            with open(yaml_configs_path, 'w', encoding='utf-8') as f:
                yaml.dump(yaml_configs, f, default_flow_style=False)
else:
    Path(f"configs/{configs.model_name}/{configs.model_id}").mkdir(parents=True, exist_ok=True)
    if yaml_configs_path_deprecated.exists():
        # migrate from deprecated folder structure to new one
        yaml_configs_path_deprecated.replace(yaml_configs_path) # will overwrite if destination exists
    else:
        # save yaml if not exist
        with open(yaml_configs_path, 'w', encoding='utf-8') as f:
            yaml.dump(asdict(configs), f, default_flow_style=False)
