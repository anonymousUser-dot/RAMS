from dataclasses import dataclass

@dataclass
class ExpConfigs:
    '''
    dataclass for argparse typo check, making life easier

    Make sure to update this dataclass after adding new args in argparse
    '''
    # basic config
    task_name: str
    is_training: int
    model_id: str
    model_name: str
    checkpoints: str
    ablation_name: str

    # dataset & data loader
    dataset_name: str
    dataset_root_path: str
    dataset_file_name: str
    features: str
    target_variable_name: str
    target_variable_index: int
    freq: str
    collate_fn: str
    augmentation_ratio: int
    missing_rate: float
    train_val_loader_shuffle: int
    train_val_loader_drop_last: int
    pin_memory: int
    persistent_workers: int
    prefetch_factor: int
    non_blocking_transfer: int
    test_inference_time: int

    # forecasting task
    seq_len: int
    label_len: int
    pred_len: int

    # classification task
    n_classes: int

    # GPU
    use_gpu: int
    gpu_id: int
    use_multi_gpu: int
    gpu_ids: str
    tf32: int
    cudnn_benchmark: int
    amp_dtype: str

    # training
    wandb: int
    sweep: int
    val_interval: int
    num_workers: int
    disable_tqdm: int
    seed_base: int
    itr: int
    train_epochs: int
    batch_size: int
    patience: int
    learning_rate: float
    loss: str
    mae_weight: float
    lr_scheduler: str
    pretrained_checkpoint_root_path: str
    pretrained_checkpoint_file_name: str
    n_train_stages: str
    retain_graph: int

    # testing
    checkpoints_test: str
    test_all: int
    test_split: str
    test_flop: int
    test_train_time: int
    test_gpu_memory: int
    test_zero_shot: int
    test_dataset_statistics: int
    save_arrays: int
    load_checkpoints_test: int

    # model configs
    # common
    patch_len: int
    patch_stride: int
    revin: int
    revin_affine: int
    kernel_size: int
    individual: int
    channel_independence: int
    scale_factor: int
    top_k: int
    embed_type: int
    enc_in: int
    dec_in: int
    c_out: int
    d_model: int
    d_timesteps: int
    n_heads: int
    n_layers: int
    e_layers: int
    d_layers: int
    hidden_layers: int
    d_ff: int
    moving_avg: int
    factor: int
    dropout: float
    embed: str
    activation: str
    output_attention: int
    node_dim: int
    # PatchTST
    patchtst_fc_dropout: float
    patchtst_head_dropout: float
    patchtst_padding_patch: str
    patchtst_subtract_last: int
    patchtst_decomposition: int
    # Mamba
    mamba_d_conv: int
    mamba_expand: int
    # Latent ODE
    latent_ode_units: int
    latent_ode_gen_layers: int
    latent_ode_rec_layers: int
    latent_ode_z0_encoder: str
    latent_ode_rec_dims: int
    latent_ode_gru_units: int
    latent_ode_classif: int
    latent_ode_linear_classif: int
    # CRU
    cru_num_basis: int
    cru_bandwidth: int
    cru_ts: float
    # NeuralFlows
    neuralflows_flow_model: str
    neuralflows_flow_layers: int
    neuralflows_latents: int
    neuralflows_time_net: str
    neuralflows_time_hidden_dim: int
    # PrimeNet
    primenet_pooling: str
    # mTAN
    mtan_num_ref_points: int
    mtan_alpha: float
    # TimeMixer
    timemixer_decomp_method: str
    timemixer_use_norm: int
    timemixer_down_sampling_layers: int
    timemixer_down_sampling_method: str
    # Nonstationary Transformer
    nonstationarytransformer_p_hidden_dims: list
    nonstationarytransformer_p_hidden_layers: int
    # Informer
    informer_distil: int
    # tPatchGNN
    tpatchgnn_te_dim: int
    # SPECTRON
    spectron_num_kernels: int
    spectron_d_max: float
    spectron_patch_len: int  # <-- 新增
    spectron_patch_stride: int  # <-- 新增
    spectron_num_intra_layers: int
    spectron_kernel_chunk_size: int
    spectron_num_last_patches: int
    # TAC-Mixer
    tac_patch_num: int
    tac_mixer_hidden_dim_p: int
    tac_mixer_hidden_dim_c: int
    tac_decoder_context_k: int
    # ASTGI
    astgi_k_neighbors: int
    astgi_prop_layers: int
    astgi_channel_dim: int
    astgi_time_dim: int
    astgi_mlp_ratio: float
    astgi_channel_dist_weight: float
    # APN
    apn_te_dim: int
    apn_npatch: int
    apn_patch_size: float
    apn_nlayer: int
    apn_attn_heads: int
    apn_research_variant: str
    apn_mcr_gate_init: float
    apn_mcr_residual_l2: float
    apn_mcr_integral_centers: int
    apn_mcr_integral_width: float
    apn_alias_remainder_scale: float
    apn_alias_remainder_l2: float
    apn_alias_score_gate_weight: float
    apn_alias_variance_price_weight: float
    apn_alias_scale_price_weight: float
    apn_lamr_motion_scale: float
    apn_lamr_correction_scale: float
    apn_diff_loss_weight: float
    apn_diff2_loss_weight: float
    apn_diff_loss_beta: float
    apn_kinematic_gate_init: float
    apn_kinematic_slope_scale: float
    apn_kinematic_accel_scale: float
    apn_kinematic_slope_clip: float
    apn_mcr_state_mode: str
    apn_mcg_gate_init: float
    apn_mcg_warmup_stage: int
    apn_mcg_warmup_blend: float
    apn_mcg_boundary_weight: float
    apn_mcg_supervised_weight: float
    apn_mcg_supervised_temperature: float
    apn_mcg_supervised_margin: float
    apn_mcg_supervised_min_confidence: float
    apn_query_count: int
    apn_varmix_init: float
    apn_fourier_init: float
    apn_reliability_source_scale: float
    apn_reliability_target_scale: float
    apn_reliability_impute_weight: float
    apn_reliability_route_weight: float
    apn_selector_hidden: int
    apn_selector_l1: float
    apn_selector_residual: float
    apn_selector_entropy: float
    apn_selector_level: str
    apn_selector_init: float
    apn_selector_scale: float
    apn_selector_safety: int
    apn_selector_safety_init: float
    apn_selector_temperature: float
    apn_selector_safety_temperature: float
    apn_selector_branch_mask: str
    apn_selector_branch_dropout: float
    apn_selector_stat_control: str
    apn_selector_uncertainty_weight: float
    apn_selector_uncertainty_mode: str
    apn_selector_trust_weight: float
    apn_selector_trust_cap: float
    apn_selector_mass_weight: float
    apn_selector_mass_min: float
    apn_selector_mass_max: float
    apn_operator_price_weight: float
    apn_operator_price_margin: float
    apn_operator_price_init: float
    apn_operator_dep_branch: str
    apn_emit_noop_pred: int
    apn_emit_selector_arrays: int
    apn_emit_mcg_arrays: int
    apn_paired_safe_weight: float
    apn_paired_safe_margin: float
    apn_gate_harm_weight: float
    apn_gate_bce_weight: float
    apn_gate_target_margin: float
    apn_paired_cvar_q: float
    apn_paired_benefit_weight: float
    apn_paired_benefit_margin: float
    apn_paired_benefit_temperature: float
    apn_paired_benefit_cap: float
    apn_paired_lcb_var_weight: float
    apn_paired_lcb_var_cap: float
    apn_paired_ratio_rho: float
    apn_paired_ratio_eps: float
    apn_paired_ratio_cap: float
    apn_paired_ntr_weight: float
    apn_paired_ntr_temperature: float
    # MCOR black-box wrapper
    mcor_base_model_name: str
    mcor_force_branch: str
    mcor_freeze_base: int
    mcor_reliability_mode: str
    # Used to be compatible with ipython. Never used
    f: int = 1

    # args not presented in argparse
    seq_len_max_irr: int = None # maximum number of observations along time dimension of x, set in irregular time series datasets
    pred_len_max_irr: int = None # maximum number of observations along time dimension of y, set in irregular time series datasets
    patch_len_max_irr: int = None # maximum number of observations along time dimension in a patch of x, set in irregular time series datasets
    subfolder_train: str = "" # timestamp of training in format %Y_%m%d_%H%M
    itr_i: int = 0 # current training iteration. [0, itr-1]
