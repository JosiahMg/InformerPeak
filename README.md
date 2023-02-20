# InformerPeak

1. 数据量少 将训练集占比增加
2. 一个模型同时训练出四个值 修改为只使用一个模型各自训练
3. 调高学习率 提前结束训练

## 最优参数
### input
seq_len=96, label_len=24, pred_len=24, 
enc_in=30, dec_in=30, c_out=1, 
d_model=512, n_heads=8, e_layers=2, 
d_layers=1, s_layers=[3, 2, 1], 
d_ff=2048, factor=1, padding=0, 
distil=True, dropout=0.05, attn='prob', 
embed='timeF', activation='gelu', 
output_attention=False, do_predict=True, 
mix=True, cols=None, num_workers=0, itr=1, 
train_epochs=10, batch_size=32, patience=5, 
learning_rate=1e-04, des='test', loss='mse', 
lradj='type1', use_amp=False, inverse=False, 
use_gpu=True, gpu=0, use_multi_gpu=False, 
devices='0,1,2,3', detail_freq='h'


### output
seq_len=96, label_len=24, pred_len=24, enc_in=30, 
dec_in=30, c_out=1, d_model=512, n_heads=8, e_layers=2, 
d_layers=1, s_layers=[3, 2, 1], d_ff=2048, factor=1, 
padding=0, distil=True, dropout=0.05, attn='prob', 
embed='timeF', activation='gelu', output_attention=False, 
do_predict=True, mix=True, cols=None, num_workers=0, itr=1, 
train_epochs=10, batch_size=32, patience=5, learning_rate=0.0001, 
des='test', loss='mse', lradj='type1', use_amp=False, 
inverse=False, use_gpu=True, gpu=0, use_multi_gpu=False,

