import torch
from torch_pesq import PesqLoss
import numpy as np

# mos = self.pesq_loss.mos(clean.squeeze(1), est_sources.squeeze(1)).mean()
# loss3 = self.pesq_loss(clean.squeeze(1), est_sources.squeeze(1)).mean()
def sisdr_loss(preds, targets, eps=1e-8):

        if preds.dim() == 3:
            preds = preds.squeeze(1)
        if targets.dim() == 3:
            targets = targets.squeeze(1)
        preds_mean = torch.mean(preds, dim=-1, keepdim=True)
        targets_mean = torch.mean(targets, dim=-1, keepdim=True)
        preds = preds - preds_mean
        targets = targets - targets_mean
        target_energy = torch.sum(targets ** 2, dim=-1, keepdim=True) + eps
        # dot = <s, s_hat>
        dot = torch.sum(preds * targets, dim=-1, keepdim=True)
        
        # alpha = <s, s_hat> / <s, s>
        alpha = dot / target_energy
        e_target = alpha * targets
        e_res = preds - e_target

        numerator = torch.sum(e_target ** 2, dim=-1) + eps
        denominator = torch.sum(e_res ** 2, dim=-1) + eps    
        sisdr = 10 * torch.log10(numerator / denominator)
        
        return -torch.mean(sisdr)


import torch
import torch.nn as nn
import auraloss

class SpectrogramMRSTFTLoss_Auraloss(nn.Module):
    def __init__(self, 
                 model_n_fft=512, model_hop_length=128, model_win_length=512,
                 device='cuda'):
        super().__init__()
        
        self.model_n_fft = model_n_fft
        self.model_hop_length = model_hop_length
        self.model_win_length = model_win_length
        self.window = torch.hann_window(model_win_length)
        self.register_buffer('istft_window', self.window)
        self.mr_stft = auraloss.freq.MultiResolutionSTFTLoss()

    def istft_wrapper(self, x):

        x_complex = torch.complex(x[:, 0], x[:, 1]) 
        x_complex = x_complex.permute(0, 2, 1)
        wav = torch.istft(
            x_complex, 
            n_fft=self.model_n_fft, 
            hop_length=self.model_hop_length, 
            win_length=self.model_win_length,
            window=self.istft_window.to(x.device),
            center=True,
            return_complex=False
        )
        return wav

    def forward(self, pred_spec, target_spec):
        pred_wav = self.istft_wrapper(pred_spec)
        if target_spec.dim() == 2:
            target_wav = target_spec
        else:
            target_wav = self.istft_wrapper(target_spec)
        
        min_len = min(pred_wav.shape[-1], target_wav.shape[-1])
        pred_wav = pred_wav[..., :min_len]
        target_wav = target_wav[..., :min_len]
        return self.mr_stft(pred_wav.unsqueeze(1), target_wav.unsqueeze(1))


def phase_losses(phase_r, phase_g):

    ip_loss = torch.mean(anti_wrapping_function(phase_r - phase_g))
    gd_loss = torch.mean(anti_wrapping_function(torch.diff(phase_r, dim=1) - torch.diff(phase_g, dim=1)))
    iaf_loss = torch.mean(anti_wrapping_function(torch.diff(phase_r, dim=2) - torch.diff(phase_g, dim=2)))

    loss_pha = ip_loss + gd_loss + iaf_loss

    return loss_pha

def anti_wrapping_function(x):

    return torch.abs(x - torch.round(x / (2 * np.pi)) * 2 * np.pi)

def loss_fn(est_ri, sigma_z, tgt_ri, z, sigma):
    # loss_all, loss_sisdr,  pesqloss, mutli_stft_loss, c_m_loss = loss_fn_p(est_ri, tgt_ri)
    pesqloss, c_m_loss = loss_fn_p(est_ri, tgt_ri)
    loss_g = loss_fn_g(sigma_z, z, sigma)
    loss = loss_g + c_m_loss  # + 0.05 * mutli_stft_loss
    return loss, pesqloss, c_m_loss, loss_g


def loss_fn_g(pred, target, sigma):
    # pred: sigma * z, target: z
    loss = torch.square(pred / sigma - target)
    loss = torch.mean(loss)
    return loss

stft_los = SpectrogramMRSTFTLoss_Auraloss(
    model_n_fft=512, 
    model_hop_length=128, 
    model_win_length=512,
)
pesq_los = PesqLoss(0.5, sample_rate=16000)
def loss_fn_p(pred, target):
    # pred: (b,2,t,f)
    stft_loss = stft_los.to(pred.device)
    pesq_loss = pesq_los.to(pred.device)
    
    wav_pred = stft_loss.istft_wrapper(pred)
    wav_target = stft_loss.istft_wrapper(target)
    # loss_sisdr = sisdr_loss(wav_pred, wav_target)
    pesqloss = pesq_loss(wav_target, wav_pred).mean()

    pred_pha = torch.atan2(pred[:, 1], pred[:, 0])
    target_pha = torch.atan2(target[:, 1], target[:, 0])

    loss_pha = phase_losses(target_pha, pred_pha)

    complex_loss = torch.mean(torch.square(pred - target))
    pred_mag = torch.sqrt(pred[:,0]**2 + pred[:,1]**2 + 1e-8)
    target_mag = torch.sqrt(target[:,0]**2 + target[:,1]**2 + 1e-8)
    mag_loss = torch.mean(torch.square(pred_mag - target_mag))
    # print(f"complex_loss: {complex_loss}, mag_loss: {mag_loss}, loss_pha: {loss_pha}")

    # mutli_stft_loss = stft_loss(pred, target)
    # loss_all = 100 * complex_loss + 100 * mag_loss + 1 * loss_sisdr + 1 * pesqloss + 1 * mutli_stft_loss

    return pesqloss, 0.5 * complex_loss + 0.5 * mag_loss + 0.002 * loss_pha



if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device: {device}")

    B, C, T, F = 4, 2, 100, 257 
    loss_fn = SpectrogramMRSTFTLoss_Auraloss(
        model_n_fft=512, 
        model_hop_length=128, 
        model_win_length=512,
        device=device
    ).to(device)

    pred_spec = torch.randn(B, C, T, F, device=device, requires_grad=True)
    target_spec = torch.randn(B, C, T, F, device=device)

    loss1 = loss_fn(pred_spec, target_spec)
    print(f"Loss: {loss1.item()}")
    
    pred_spec_2 = torch.randn(B, C, T, F, device=device, requires_grad=True)
    target_wav = torch.randn(B, 12800, device=device) 

    loss2 = loss_fn(pred_spec_2, target_wav)
    print(f"Loss: {loss2.item()}")
    
    loss2.backward()
    with torch.no_grad():
        wav_debug = loss_fn.istft_wrapper(pred_spec)
    