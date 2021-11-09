from enums import DifficultyScorerStrategy


class Config:
    def __init__(self):
        self.pickle_path = "dataset"
        self.swat_path = "/Users/xuqh/Documents/workspace/dataset/SWaT_Dataset_Attack_v0.xlsx"
        self.swat_idx_path = "dataset/swat_idx.pkl"
        self.swat_idx_indices_path = "dataset/swat_idx_indices.pkl"
        self.swat_attack_log_path = "dataset/swat_attack_log.pkl"
        self.batadal_path = "/Users/xuqh/Documents/workspace/dataset/BATADAL/BATADAL_dataset03.csv"
        self.batadal_idx_indices_path = "dataset/batadal_idx_indices.pkl"
        self.batadal_idx_path = "dataset/batadal_idx.pkl"
        self.batadal_attack_log_path = "dataset/batadal_attack_log.pkl"
        self.wadi_path = "/Users/xuqh/Documents/workspace/dataset/WADI/WADI.A1_9 Oct 2017/WADI_attackdata.csv"
        self.wadi_idx_path = "dataset/wadi_idx.pkl"
        self.wadi_idx_indices_path = "dataset/wadi_idx_indices.pkl"
        self.wadi_attack_log_path = "dataset/wadi_attack_log.pkl"
        self.swat_valid_pkl_path = "dataset/swat_valid.pkl"
        self.wadi_valid_pkl_path = "dataset/wadi_valid.pkl"
        self.batadal_valid_pkl_path = "dataset/batadal_valid.pkl"
        self.phm_path = "/Users/xuqh/Documents/workspace/dataset/phm/phm_combined.csv"
        self.phm_train_pkl_path = "dataset/phm_train.pkl"
        self.phm_test_pkl_path = "dataset/phm_test.pkl"
        self.phm_attack_log_path = "dataset/phm_attack_log.pkl"
        self.gas_path = "/Users/xuqh/Documents/workspace/dataset/gas/water_final.arff"
        self.gas_valid_pkl_path = "dataset/gas_valid.pkl"
        self.gas_train_pkl_path = "dataset/gas_train.pkl"
        self.gas_test_pkl_path = "dataset/gas_test.pkl"
        self.gas_attack_log_path = "dataset/gas_attack_log.pkl"
        self.swat_graph = "dataset/swat_graph.pkl"
        self.wadi_graph = "dataset/wadi_graph.pkl"
        self.batadal_graph = "dataset/batadal_graph.pkl"
        self.model_path = "saved"
        self.slow_start = 1000
        self.pretrained = ""
        self.hybrid_difficulty_ratio = 0.2
        # model configs
        self.classification_size = 2
        self.batch_size = 64
        self.window_size = self.batch_size
        self.hidden_size = 100
        self.cnn_kernel_size = 1
        self.cnn_out_channels = 32
        self.train_test_ratio = 0.8
        self.n_epochs = 1000
        self.threshhold = 0.64
        self.repeat = 30
        self.cnn_in_channels = 128
        self.pool_kernel_size = (4, 2)
        self.latent_X_shape = (
            1, 1, self.batch_size * self.pool_kernel_size[0], self.hidden_size * self.pool_kernel_size[1])
        self.gan_lbd = 0.64
        self.cusum_threshhold = 0.8
        self.gan_generator_out_size = 64
        self.dropout = 0.2
        self.hmt = 0.8
        self.difficulty_scorer_strategy = DifficultyScorerStrategy.HYBRID
        self.dataset_min = 0
        self.dataset_max = 10000
        self.baby_step_ratio=0.8
