import yaml
import os


class Config:

    def __init__(self, cfg_id, test=False):
        self.id = cfg_id
        cfg_name = 'motion_pred/cfg/%s.yml' % cfg_id
        if not os.path.exists(cfg_name):
            print("Config file doesn't exist: %s" % cfg_name)
            exit(0)
        cfg = yaml.safe_load(open(cfg_name, 'r'))

        # create dirs
        self.base_dir = '/tmp' if test else 'results'

        self.cfg_dir = '%s/%s' % (self.base_dir, cfg_id)
        self.model_dir = '%s/models' % self.cfg_dir
        self.result_dir = '%s/results' % self.cfg_dir
        self.log_dir = '%s/log' % self.cfg_dir
        self.tb_dir = '%s/tb' % self.cfg_dir
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.tb_dir, exist_ok=True)

        # common
        self.dataset = cfg.get('dataset', 'h36m')
        self.batch_size = cfg.get('batch_size', 8)
        self.normalize_data = cfg.get('normalize_data', False)
        self.save_model_interval = cfg.get('save_model_interval', 20)
        self.t_his = cfg['t_his']
        self.t_pred = cfg['t_pred']
        self.use_vel = cfg.get('use_vel', False)

        # vae
        self.nz = cfg['nz']
        self.beta = cfg['beta']
        self.omega = cfg['omega']
        self.lambda_v = cfg.get('lambda_v', 0)
        self.vae_lr = cfg['vae_lr']
        self.vae_specs = cfg.get('vae_specs', dict())
        self.num_vae_epoch = cfg['num_vae_epoch']
        self.num_vae_epoch_fix = cfg.get('num_vae_epoch_fix', self.num_vae_epoch)
        self.num_vae_data_sample = cfg['num_vae_data_sample']
        self.vae_model_path = os.path.join(self.model_dir , 'vae_%04d.p')
        self.nk = cfg.get('nk', 5)
        self.nj = cfg.get('nj', 23)
        
