
# these parameters are all you need. we pass this class to various functions
class ModelParams():
    def __init__(self):
        self.d_model = 512
        self.num_heads = 8
        self.n_decoders = 6
        self.n_encoders = 6
        self.d_v = 0
        self.d_k = 0
        self.d_ff = 0
        self.train_steps = 0
        self.dropout = 0.1
        self.adam_params = {
            'beta_1': 0.9,
            'beta_2': 0.98,
            'epsilon': 10e-9
        }
