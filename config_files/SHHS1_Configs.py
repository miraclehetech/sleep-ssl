class Config(object):
    def __init__(self):
        # model configs
        self.channel='Resp'
        self.input_channels = 1
        self.record='30seconds'
        if self.channel!='Resp':
            self.kernel_size = 25
            # self.stride=5
            self.stride = 6
        else:
            self.kernel_size = 15
            self.stride = 2

        self.num_classes = 2
        self.embed_dim = 256
        
        # training configs
        self.num_epoch = 100
        self.mutiple_models=True
        self.channel_list=['ECG','EEG','Resp']
        self.lr = 5e-5
        self.eta_min = 1e-5
        
        # data parameters
        self.drop_last = True
        self.batch_size1 = 500
        self.batch_size2=1
        self.old_model=False
        self.augmentation = augmentations()


class augmentations(object):
    def __init__(self):
        self.jitter_scale_ratio = 0.001
        self.jitter_ratio = 0.01
        self.max_seg = 5
