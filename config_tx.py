from sklearn import preprocessing
from config import PREDICT_LEN

class data_config(object):
    max_limit=-1
    batch_size=200
    train_pre=0.8
    valid_pre=0.05
    DATA_PATH="./datasets/tx.xlsx"
    PEARSON_PATH="./datasets/tx_pearson.xlsx"
    seq_len=12
    predict_len=PREDICT_LEN
    minMaxScaler=preprocessing.MinMaxScaler((-1,1))


class model_config(object):
    SEED=2
    epoches = 60
    lr = 0.01
    APS_LSTM_Args={
        'num_nodes':12,
        'seq_len':12,
        'pred_len':6,
        'hidden_size': 82,
        'n_layers':2, 
        'top_k': 2,
        'qkv_bias': False,
        'attn_drop':0.,
        'proj_drop':0.,
        'lr':0.01,
        'use_pa':True,
        'use_sa':True,
        'adj_mx': None
    }


    
