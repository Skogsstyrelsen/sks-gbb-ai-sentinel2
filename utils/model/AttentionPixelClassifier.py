from utils.model.AttentionBlock import *
import torch


class AttentionCENet(nn.Module):
    def __init__(self, input_numChannels, output_numChannels, dropout_prob = 0.0):
        super(AttentionCENet, self).__init__()

        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = ReluConvBlock(input_numChannels, 32, dropout_prob)
        self.Conv2 = ReluConvBlock(32, 64, dropout_prob)
        self.Conv3 = ReluConvBlock(64, 128, dropout_prob)
        self.Conv4 = ReluConvBlock(128, 256, dropout_prob)

        self.Up4 = ReluUpConv(int(512//2), int(256/2), dropout_prob)
        self.Att4 = AttentionBlock(F_g=int(256//2), F_l=int(256//2), n_coefficients=int(128//2))
        self.UpConv4 = ReluConvBlock(int(512//2), int(256//2), dropout_prob)

        self.Up3 = ReluUpConv(int(256//2), int(128//2), dropout_prob)
        self.Att3 = AttentionBlock(F_g=int(128//2), F_l=int(128//2), n_coefficients=int(64//2))
        self.UpConv3 = ReluConvBlock(int(256//2), int(128//2), dropout_prob)

        self.Up2 = ReluUpConv(int(128//2), int(64//2), dropout_prob)
        self.Att2 = AttentionBlock(F_g=int(64//2), F_l=int(64//2), n_coefficients=int(32//2))
        self.UpConv2 = ReluConvBlock(int(128//2), int(64//2), dropout_prob)

        self.Conv = nn.Conv2d(int(64//2), output_numChannels, kernel_size=1, stride=1, padding=0)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        e : encoder layers
        d : decoder layers
        s : skip-connections from encoder layers to decoder layers
        """
        e1 = self.Conv1(x)

        e2 = self.MaxPool(e1)
        e2 = self.Conv2(e2)

        e3 = self.MaxPool(e2)
        e3 = self.Conv3(e3)

        e4 = self.MaxPool(e3)
        e4 = self.Conv4(e4)


        d4 = self.Up4(e4)

        s3 = self.Att4(gate=d4, skip_connection=e3)
        d4 = torch.cat((s3, d4), dim=1)
        d4 = self.UpConv4(d4)

        d3 = self.Up3(d4)
        s2 = self.Att3(gate=d3, skip_connection=e2)
        d3 = torch.cat((s2, d3), dim=1)
        d3 = self.UpConv3(d3)

        d2 = self.Up2(d3)
        s1 = self.Att2(gate=d2, skip_connection=e1)
        d2 = torch.cat((s1, d2), dim=1)
        d2 = self.UpConv2(d2)

        out = self.Conv(d2)
        out = self.sigmoid(out)
        
        return out
    