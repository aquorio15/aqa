import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from cross_attention import CrossAttention

def exists(val):
    return val is not None

class FC(nn.Module):
    def __init__(self, in_size, out_size, dropout_r=0.0, use_relu=True):
        super(FC, self).__init__()
        self.dropout_r = dropout_r
        self.use_relu = use_relu

        self.linear = nn.Linear(in_size, out_size)

        if use_relu:
            self.relu = nn.ReLU(inplace=True)

        if dropout_r > 0:
            self.dropout = nn.Dropout(dropout_r)

    def forward(self, x):
        x = self.linear(x)

        if self.use_relu:
            x = self.relu(x)

        if self.dropout_r > 0:
            x = self.dropout(x)

        return x


class MLP(nn.Module):
    def __init__(self, in_size, mid_size, out_size, dropout_r=0.0, use_relu=True):
        super(MLP, self).__init__()

        self.fc = FC(in_size, mid_size, dropout_r=dropout_r, use_relu=use_relu)
        self.linear = nn.Linear(mid_size, out_size)

    def forward(self, x):
        return self.linear(self.fc(x))


class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps

        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class MHAtt(nn.Module):
    def __init__(self, hidden_size=512, dropout=0.1, n_head=8):
        super(MHAtt, self).__init__()

        self.hidden_size = hidden_size
        self.dropout = dropout
        self.n_head = n_head
        self.linear_v = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_k = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_q = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_merge = nn.Linear(self.hidden_size, self.hidden_size)

        self.dropout = nn.Dropout(self.dropout)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)

        v = (
            self.linear_v(v)
            .view(n_batches, -1, self.n_head, int(self.hidden_size / self.n_head))
            .transpose(1, 2)
        )

        k = (
            self.linear_k(k)
            .view(n_batches, -1, self.n_head, int(self.hidden_size / self.n_head))
            .transpose(1, 2)
        )

        q = (
            self.linear_q(q)
            .view(n_batches, -1, self.n_head, int(self.hidden_size / self.n_head))
            .transpose(1, 2)
        )

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(n_batches, -1, self.hidden_size)

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)


class FFN(nn.Module):
    def __init__(self, hidden_size=512, dropout=0.1, ff_dim=2048):
        super(FFN, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.ff_dim = ff_dim
        self.mlp = MLP(
            in_size=self.hidden_size,
            mid_size=self.ff_dim,
            out_size=self.hidden_size,
            dropout_r=self.dropout,
            use_relu=True,
        )

    def forward(self, x):
        return self.mlp(x)


class SA(nn.Module):
    def __init__(self, hidden_size=512, dropout=0.1, ff_dim=2048, n_head=8):
        super(SA, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.ff_dim = ff_dim
        self.n_head = n_head

        self.mhatt = MHAtt(
            hidden_size=self.hidden_size, dropout=self.dropout, n_head=self.n_head
        )
        self.ffn = FFN(
            hidden_size=self.hidden_size, dropout=self.dropout, ff_dim=self.ff_dim
        )

        self.dropout1 = nn.Dropout(self.dropout)
        self.norm1 = LayerNorm(self.hidden_size)

        self.dropout2 = nn.Dropout(self.dropout)
        self.norm2 = LayerNorm(self.hidden_size)

    def forward(self, y, y_mask):
        y = self.norm1(y + self.dropout1(self.mhatt(y, y, y, y_mask)))
        y = self.norm2(y + self.dropout2(self.ffn(y)))
        return y


class Crossattention(nn.Module):
    def __init__(
        self,
        hidden_size_image=2048,
        dropout=0.1,
        ff_dim=2048,
        hidden_size_audio=1920,
        n_head=8,
    ):
        super(Crossattention, self).__init__()
        self.hidden_size_image = hidden_size_image
        self.dropout = dropout
        self.ff_dim = ff_dim
        self.hidden_size_audio = hidden_size_audio
        self.n_head = n_head
        self.cross_attn = CrossAttention(
            dim=self.hidden_size_image,
            heads=self.n_head,
            dim_head=self.ff_dim,
            context_dim=self.hidden_size_audio,
        )

        self.dropout1 = nn.Dropout(self.dropout)
        self.norm1 = LayerNorm(self.hidden_size_image)

        self.dropout2 = nn.Dropout(self.dropout)
        self.norm2 = LayerNorm(self.hidden_size_audio)

    def forward(self, image, audio, image_mask=None, audio_mask=None):

        #if not exists(image_mask) or exists(audio_mask):
        image_out, audio_out = self.cross_attn(image, audio)
        image_out = self.norm1(image_out + self.dropout2(image_out))
        audio_out = self.norm2(audio_out + self.dropout2(audio_out))
        return (image_out, audio_out)


class NoSA(nn.Module):  ## Model-1
    def __init__(
        self,
        hidden_size_image=2048,
        dropout=0.1,
        head_dim=2048,
        hidden_size_audio=1920,
        n_head=8,
        n_layers=6,
        out_size1=1024,
        out_size2=512,
        out_size3=256,
    ):
        super(NoSA, self).__init__()
        self.hidden_size_image = hidden_size_image
        self.dropout = dropout
        self.head_dim = head_dim
        self.hidden_size_audio = hidden_size_audio
        self.n_head = n_head
        self.n_layers = n_layers

        self.linear1 = nn.Linear(hidden_size_audio, out_size1)
        self.linear2 = nn.Linear(hidden_size_image, out_size1)
        self.linear3 = nn.Linear(out_size1, out_size2)
        self.linear4 = nn.Linear(out_size2, out_size3)

        self.enc_list = nn.ModuleList(
            [
                Crossattention(
                    hidden_size_image=self.hidden_size_image,
                    dropout=self.dropout,
                    ff_dim=self.head_dim,
                    hidden_size_audio=self.hidden_size_audio,
                    n_head=self.n_head,
                )
                for _ in range(self.n_layers)
            ]
        )

    def forward(self, image, audio, image_mask=None, audio_mask=None):
        for enc in self.enc_list:
            image_out, audio_out = enc(
                image, audio
            )
        audio_out1 = self.linear1(audio_out)
        audio_out2 = self.linear3(audio_out1)
        audio_out3 = self.linear4(audio_out2)
        image_out1 = self.linear2(image_out)
        image_out2 = self.linear3(image_out1)
        image_out3 = self.linear4(image_out2)
        return image_out3, audio_out3


class SA1(nn.Module):  ## Model-2
    def __init__(
        self,
        hidden_size_image=2048,
        dropout=0.1,
        head_dim=2048,
        hidden_size_audio=1920,
        n_head=8,
        n_layers=6,
        out_size1=1024,
        out_size2=512,
        out_size3=256,
    ):
        super(SA1, self).__init__()
        self.hidden_size_image = hidden_size_image
        self.dropout = dropout
        self.head_dim = head_dim
        self.hidden_size_audio = hidden_size_audio
        self.n_head = n_head
        self.n_layers = n_layers

        self.dec_list = nn.ModuleList(
            [
                Crossattention(
                    hidden_size_image=self.hidden_size_image,
                    dropout=self.dropout,
                    ff_dim=self.head_dim,
                    hidden_size_audio=self.hidden_size_audio,
                    n_head=self.n_head,
                )
                for _ in range(self.n_layers)
            ]
        )
        self.enc_list = nn.ModuleList(
            [
                SA(
                    hidden_size=self.hidden_size_audio,
                    dropout=self.dropout,
                    ff_dim=self.head_dim,
                    n_head=self.n_head,
                )
                for _ in range(self.n_layers)
            ]
        )

        self.linear1 = nn.Linear(hidden_size_audio, out_size1)
        self.linear2 = nn.Linear(hidden_size_image, out_size1)
        self.linear3 = nn.Linear(out_size1, out_size2)
        self.linear4 = nn.Linear(out_size2, out_size3)

    def forward(self, image, audio, image_mask=None, audio_mask=None):

        for enc in self.enc_list:
            y = enc(audio, audio_mask)

        for dec in self.dec_list:
            image_out, audio_out = dec(
                image, y, image_mask=image_mask, audio_mask=audio_mask
            )

        audio_out1 = self.linear1(audio_out)
        audio_out2 = self.linear3(audio_out1)
        audio_out3 = self.linear4(audio_out2)
        image_out1 = self.linear2(image_out)
        image_out2 = self.linear3(image_out1)
        image_out3 = self.linear4(image_out2)

        return image_out3, audio_out3


class SA2(nn.Module):
    def __init__(
        self,
        hidden_size_image=2048,
        dropout=0.1,
        head_dim=2048,
        hidden_size_audio=1920,
        n_head=8,
    ):
        super(SA2, self).__init__()

        self.hidden_size_image = hidden_size_image
        self.dropout = dropout
        self.head_dim = head_dim
        self.hidden_size_audio = hidden_size_audio
        self.n_head = n_head

        self.cross = Crossattention(
            hidden_size_image=self.hidden_size_image,
            dropout=self.dropout,
            ff_dim=self.head_dim,
            hidden_size_audio=self.hidden_size_audio,
            n_head=self.n_head,
        )
        self.sa = SA(
            hidden_size=self.hidden_size_image,
            dropout=self.dropout,
            ff_dim=self.head_dim,
            n_head=self.n_head,
        )

    def forward(self, image, audio, image_mask=None, audio_mask=None):

        y = self.sa(image, image_mask)
        image_out, audio_out = self.cross(
            y, audio, image_mask=image_mask, audio_mask=audio_mask
        )
        return image_out, audio_out


class SA3(nn.Module):  ## Model-3
    def __init__(
        self,
        hidden_size_image=2048,
        dropout=0.1,
        head_dim=2048,
        hidden_size_audio=1920,
        n_head=8,
        n_layers=6,
        out_size1=1024,
        out_size2=512,
        out_size3=256,
    ):
        super(SA3, self).__init__()
        self.hidden_size_image = hidden_size_image
        self.dropout = dropout
        self.head_dim = head_dim
        self.hidden_size_audio = hidden_size_audio
        self.n_head = n_head
        self.n_layers = n_layers
        self.dec_list = nn.ModuleList(
            [
                SA2(
                    hidden_size_image=self.hidden_size_image,
                    n_head=self.n_head,
                    dropout=self.dropout,
                    head_dim=self.head_dim,
                    hidden_size_audio=self.hidden_size_audio,
                )
                for _ in range(self.n_layers)
            ]
        )
        self.enc_list = nn.ModuleList(
            [
                SA(
                    hidden_size=self.hidden_size_audio,
                    dropout=self.dropout,
                    ff_dim=self.head_dim,
                    n_head=self.n_head,
                )
                for _ in range(self.n_layers)
            ]
        )

        self.linear1 = nn.Linear(hidden_size_audio, out_size1)
        self.linear2 = nn.Linear(hidden_size_image, out_size1)
        self.linear3 = nn.Linear(out_size1, out_size2)
        self.linear4 = nn.Linear(out_size2, out_size3)

    def forward(self, image, audio, image_mask=None, audio_mask=None):

        for enc in self.enc_list:
            y = enc(audio, audio_mask)
        for dec in self.dec_list:
            image_out, audio_out = dec(
                image, y, image_mask=image_mask, audio_mask=audio_mask
            )

        audio_out1 = self.linear1(audio_out)
        audio_out2 = self.linear3(audio_out1)
        audio_out3 = self.linear4(audio_out2)
        image_out1 = self.linear2(image_out)
        image_out2 = self.linear3(image_out1)
        image_out3 = self.linear4(image_out2)

        return image_out3, audio_out3


# NOTE: checking the models

# a = SA3()
# image = torch.randn(1, 40, 2048)
# audio = torch.randn(1, 120, 1920)

# image_mask = torch.ones((1, 40)).bool()
# audio_mask = torch.ones((1, 120)).bool()
# b, c = a(image, audio, image_mask=image_mask, audio_mask=audio_mask)
# print(b.shape)
# print(c.shape)
