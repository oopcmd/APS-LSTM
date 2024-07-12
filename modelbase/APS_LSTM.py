import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from scipy import sparse as sp
import numpy as np


def _calculate_normalized_laplacian(adj):
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    isolated_point_num = np.sum(np.where(d, 0, 1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian, isolated_point_num

def _cal_lape(adj_mx, lape_dim):
    L, isolated_point_num = _calculate_normalized_laplacian(adj_mx)
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort()
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
    laplacian_pe = torch.from_numpy(EigVec[:, isolated_point_num + 1: lape_dim + isolated_point_num + 1]).float()
    laplacian_pe.require_grad = False
    return laplacian_pe

class data_embedding(nn.Module):
    def __init__(self,adj_mx,lape_dim) -> None:
        super().__init__()
        self.adj_mx=_cal_lape(adj_mx,lape_dim)
        self.lp_enc=nn.Linear(lape_dim,1)
    
    def forward(self,x):
        if self.adj_mx.device != x.device:
            self.adj_mx = self.adj_mx.to(x.device)
        return x+self.lp_enc(self.adj_mx).squeeze(-1)


def FFT_for_Period(x, k=2):
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class PeriodAttention(nn.Module):
    def __init__(self, config:dict):
        super().__init__()
        num_nodes=config['num_nodes']
        qkv_bias=config.get('qkv_bias',False)
        attn_drop=config.get('attn_drop',0.)
        proj_drop=config.get('proj_drop',0.)

        self.p_q_conv = nn.Conv2d(num_nodes, num_nodes, kernel_size=1, bias=qkv_bias)
        self.p_k_conv = nn.Conv2d(num_nodes, num_nodes, kernel_size=1, bias=qkv_bias)
        self.p_v_conv = nn.Conv2d(num_nodes, num_nodes, kernel_size=1, bias=qkv_bias)
        self.p_attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(num_nodes, num_nodes)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x):
        B, N, P, F = x.shape
        p_q = self.p_q_conv(x)
        p_k = self.p_k_conv(x)
        p_v = self.p_v_conv(x)

        p_attn = (p_q @ p_k.transpose(-2, -1)) * (F**-0.5) 

        p_attn = p_attn.softmax(dim=-1)
        p_attn = self.p_attn_drop(p_attn)
        p_x = (p_attn @ p_v).transpose(1, -1)

        x = self.proj(p_x)
        x = self.proj_drop(x)
        return x.transpose(1,-1)
    


class SpatialAttention(nn.Module):
    def __init__(self,config:dict) -> None:
        super().__init__()
        num_nodes=config['num_nodes']
        qkv_bias=config.get('qkv_bias',False)
        attn_drop=config.get('attn_drop',0.)
        proj_drop=config.get('proj_drop',0.)

        self.s_q_conv = nn.Conv1d(num_nodes, num_nodes, kernel_size=1, bias=qkv_bias)
        self.s_k_conv = nn.Conv1d(num_nodes, num_nodes, kernel_size=1, bias=qkv_bias)
        self.s_v_conv = nn.Conv1d(num_nodes, num_nodes, kernel_size=1, bias=qkv_bias)
        self.s_attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(num_nodes, num_nodes)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, T, N = x.shape
        s_q = self.s_q_conv(x.transpose(-1,-2))
        s_k = self.s_k_conv(x.transpose(-1,-2))
        s_v = self.s_v_conv(x.transpose(-1,-2))

        s_attn = (s_q @ s_k.transpose(-1, -2)) * (N**-0.5) 

        s_attn = s_attn.softmax(dim=-1)
        s_attn = self.s_attn_drop(s_attn)
        p_x = (s_attn @ s_v).transpose(-1, -2)

        x = self.proj(p_x)
        x = self.proj_drop(x)
        return x


class PSBlock(nn.Module):
    def __init__(self, config:dict):
        super().__init__()
        self.k = config.get('top_k',2)
        self.use_pa=config.get('use_pa',True)
        self.use_sa=config.get('use_sa',True)
        self.period=nn.ModuleList([PeriodAttention(config) for i in range(self.k)])
        self.spatial=nn.ModuleList([SpatialAttention(config) for i in range(self.k)])
        
    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i, period in zip(range(self.k),period_list):
            if T % period != 0:
                length = ((T // period) + 1) * period
                out = F.pad(x,[0,0,0,length-T,0,0])
            else:
                length = T
                out = x
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            if self.use_pa:
                out = self.period[i](out)
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)[:, :T, :]
            if self.use_sa:
                out = self.spatial[i](out)
            res.append(out)
        res = torch.stack(res, dim=-1)
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        res = res + x
        return res
    
  
class APS_LSTM(nn.Module):
    def __init__(self,config:dict) -> None:
        super().__init__()
        num_nodes=config['num_nodes']
        n_layers=config.get('n_layers',2)
        pred_len=config.get('pred_len',6)
        hidden_size= config.get('hidden_size',32)
        lape_dim=config.get('lape_dim',num_nodes//2)
        adj_mx=config['adj_mx']
        self.embed=data_embedding(adj_mx,lape_dim)
        self.ps_layers=nn.Sequential()
        for _ in range(n_layers):
            self.ps_layers.append(PSBlock(config))
        self.lstm=nn.LSTM(input_size=num_nodes,hidden_size=hidden_size,num_layers=1,batch_first=True)
        self.fc_out=nn.Linear(hidden_size,pred_len)

    def forward(self,x):   
        x=self.embed(x)     
        x=self.ps_layers(x)
        out, _ = self.lstm(x)
        return self.fc_out(out[:,-1,:])


