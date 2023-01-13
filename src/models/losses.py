import torch
from torch import nn

class TacotronLoss(nn.Module):
    def __init__(
        self,
    ):
        super(TacotronLoss, self).__init__()
    
    @staticmethod
    def get_masked_loss(mel, mel_target, mel_masks, mel_lens):
        """masked the padded place
        """
        mel_loss = nn.MSELoss(reduction='none')(mel, mel_target).mean(-1) # shape=(B, len_mel)
        mel_loss = mel_loss.masked_fill_(~mel_masks, 0.)
        mel_loss = mel_loss.sum()/mel_lens.sum()
        return mel_loss

    def forward(
        self,
        postnet_outputs,
        decoder_outputs,
        mel_masks,
        mel_lens,
        stop_tokens,
        mel_targets,
        stop_targets,

    ):
        decoder_loss = self.get_masked_loss(decoder_outputs, mel_targets, mel_masks, mel_lens)#nn.MSELoss(reduction='none')(decoder_outputs, mel_targets)
        postnet_loss = self.get_masked_loss(postnet_outputs, mel_targets, mel_masks, mel_lens)#nn.MSELoss(reduction='none')(postnet_outputs, mel_targets)

        stop_targets = stop_targets.view(-1, 1)
        stop_tokens = stop_tokens.view(-1, 1)
        stop_loss = nn.BCEWithLogitsLoss()(stop_tokens, stop_targets)


        return {
            'loss': decoder_loss + postnet_loss + stop_loss,
            'decoder_loss': decoder_loss,
            'postnet_loss': postnet_loss,
            'stop_loss': stop_loss
        }