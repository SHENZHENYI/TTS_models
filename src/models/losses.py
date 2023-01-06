from torch import nn

class TacotronLoss(nn.Module):
    def __init__(
        self,
    ):
        super(TacotronLoss, self).__init__()
    
    def forward(
        self,
        postnet_outputs,
        decoder_outputs,
        stop_tokens,
        mel_targets,
        stop_targets,

    ):
        decoder_loss = nn.MSELoss()(decoder_outputs, mel_targets)
        postnet_loss = nn.MSELoss()(postnet_outputs, mel_targets)
        print('stop_tokens', stop_tokens.shape)
        print('stop_targets', stop_targets.shape)

        stop_targets = stop_targets.view(-1, 1)
        stop_tokens = stop_tokens.view(-1, 1)
        stop_loss = nn.BCEWithLogitsLoss()(stop_tokens, stop_targets)


        return {
            'loss': decoder_loss + postnet_loss + stop_loss,
            'decoder_loss': decoder_loss,
            'postnet_loss': postnet_loss,
            'stop_loss': stop_loss
        }