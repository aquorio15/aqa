from conv1d import *
from conv2d import *
from mamba import *
from mha import *
from mamba_simple import Mamba
import math
import sys
import torch
import torch.nn as nn
from kan_linear import *
from cross_attention import *
from self_attention import *
import torch.nn.functional as F
from einops import rearrange
# from timm.layers.helpers import to_ntuple, to_2tuple
# from timm.layers.weight_init import trunc_normal_
# from timm.models.vision_transformer import _cfg
# from timm.models.registry import register_model
from torchsummary import summary
sys.path.append('/nfsshare/Amartya/EMNLP-WACV/code_mamba/pytorch-image-models')

# %%
def exists(val):
    return val is not None


class FC(nn.Module):
    def __init__(
        self, in_size, out_size, dropout_r=False, use_relu=True, linear_type="kan"
    ):
        super(FC, self).__init__()
        self.dropout_r = dropout_r
        self.use_relu = use_relu
        self.linear_type = linear_type

        if self.linear_type == "kan":
            self.linear = FastKANLayer(in_size, out_size)
        else:
            self.linear = nn.Linear(in_size, out_size)

        if self.use_relu:
            self.relu = nn.ReLU(inplace=True)

        self.dropout = nn.Dropout(0.0)

    def forward(self, x):
        x = self.linear(x)

        if self.use_relu:
            x = self.relu(x)

        if self.dropout_r:
            x = self.dropout(x)

        return x


class MLP(nn.Module):
    def __init__(
        self,
        in_size,
        mid_size,
        out_size,
        dropout_r=0.0,
        use_relu=True,
        linear_type="kan",
    ):
        super(MLP, self).__init__()

        self.fc = FC(in_size, mid_size, dropout_r=dropout_r, use_relu=use_relu, linear_type=linear_type)
        if linear_type == "kan":
            self.linear = FastKANLayer(mid_size, out_size)
        else:
            self.linear = nn.Linear(mid_size, out_size)

    def forward(self, x):
        return self.linear(self.fc(x))


class AttFlat(nn.Module):
    def __init__(
        self,
        hidden_size=32,
        flat_mlp_size=32,
        flat_glimpses=1,
        dropout=0.1,
        flat_out_size=32,
        linear_type="kan",
    ):
        super(AttFlat, self).__init__()

        self.hidden_size = hidden_size
        self.flat_mlp_size = flat_mlp_size
        self.flat_glimpses = flat_glimpses
        self.dropout = dropout
        self.flat_out_size = flat_out_size
        self.linear_type = linear_type

        self.mlp = MLP(
            in_size=self.hidden_size,
            mid_size=self.flat_mlp_size,
            out_size=self.flat_glimpses,
            dropout_r=self.dropout,
            use_relu=True,
            linear_type=linear_type
        )
        if self.linear_type == "kan":
            self.linear_merge = FastKANLayer(
                self.hidden_size * self.flat_glimpses, self.flat_out_size
            )
        else:
            self.linear_merge = nn.Linear(
                self.hidden_size * self.flat_glimpses, self.flat_out_size
            )

    def forward(self, x):
        att = self.mlp(x)
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.flat_glimpses):
            att_list.append(torch.sum(att[:, :, i : i + 1] * x, dim=1))

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted


class MambaFormer(nn.Module):

    def __init__(
        self,
        patch_size=10,
        stride=4,
        in_chans=3,
        embed_dim=256,
        norm_layer=None,
        flatten=True,
        conv_layers=[(192, 30, 5)] + [(192, 1, 2)] * 5 + [(192, 26, 1)],
        conv_layers_text=[(256, 94, 3)],
        dropout=0.0,
        drop_path_rate=0.0,
        mode="layer_norm",
        conv_bias=False,
        depths=[2, 2, 8, 4],
        mlp_ratio=4,
        drop_rate=0.1,
        n_layers=6,
        n_heads=2,
        ape=False,
        fused_out_size=32,
        head_dim=2048,
        n_head=8,
        conv_out=32,
        mid_channels=1024,
        out_size1=1024,
        out_size2=512,
        out_size3=256,
        hidden_mlp_size=32,
        classifier=1000,
        linear_type="kan",
    ):
        super(MambaFormer, self).__init__()

        self.norm_layer = norm_layer
        self.patch_size = patch_size
        self.stride = stride
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.flatten = flatten
        self.head_dim = head_dim
        self.dropout = dropout
        self.conv_layers_text = conv_layers_text
        self.n_head = n_head
        self.conv_layers = conv_layers
        self.conv_bias = conv_bias
        self.drop_rate = drop_rate
        self.n_layers = n_layers
        self.mid_channels = mid_channels
        self.fused_out_size = fused_out_size
        self.hidden_mlp_size = hidden_mlp_size
        self.conv_out = conv_out
        self.n_heads = n_heads
        self.out_size1 = out_size1
        self.out_size2 = out_size2
        self.out_size3 = out_size3
        self.classifier = classifier
        self.img_range = 1.0
        self.embed_dim_temp = int(embed_dim / 2)
        self.linear_type = linear_type
        cur = 0
        if self.in_chans == 3 or self.in_chans == 6:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            rgbrgb_mean = (0.4488, 0.4371, 0.4040, 0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
            self.mean_in = torch.Tensor(rgbrgb_mean).view(1, 6, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)

        self.patch_embed = PatchEmbed(
            patch_size=self.patch_size,
            stride=self.patch_size,
            in_chans=self.embed_dim,
            embed_dim=self.embed_dim,
            norm_layer=nn.LayerNorm if self.norm_layer else None,
            flatten=True,
        )
        self.in_channel = 256
        self.base_dim = 256
        self.norm = nn.BatchNorm1d(2048)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.feature_extractor_audio = ConvFeatureExtractionModel(
            conv_layers=self.conv_layers,
            dropout=self.dropout,
            conv_bias=self.conv_bias,
        )
        self.feature_extractor_text = ConvFeatureExtractionModel(
            conv_layers=self.conv_layers_text,
            dropout=self.dropout,
            conv_bias=self.conv_bias,
        )
        self.attflat = AttFlat(
            hidden_size=64,
            flat_mlp_size=64,
            flat_glimpses=1,
            dropout=0.1,
            flat_out_size=32,
            linear_type=self.linear_type
        )
        self.softmax = nn.Softmax(dim=0)
        # if self.ape:
        #     self.absolute_pos_embed = nn.Parameter(
        #         torch.zeros(1, num_patches, embed_dim)
        #     )
        #     trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=self.drop_rate)
        self.apply(self._init_weights)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        ######################################################### NOTE: MAMBA feature extraction ################################################

        # self.high_level_feature_extraction = nn.Sequential(
        #     *[SingleMambaBlock(self.embed_dim) for i in range(self.n_layers)]
        # )
        ######################################################### NOTE: Shallow feature extraction ###############################################

        self.shallow_feature_extraction = nn.Sequential(
            nn.Conv2d(self.in_chans, self.embed_dim, 3, 1, 1),
            ResBlock(self.embed_dim, self.embed_dim),
            ResBlock(self.embed_dim, self.embed_dim),
            ResBlock(self.embed_dim, 3),
        )

        self.low_level_feature_extraction1 = nn.Conv2d(
            1, self.embed_dim_temp, 3, 1, 1
        )
        self.low_level_feature_extraction2 = nn.Conv2d(
            self.embed_dim_temp, self.embed_dim, 3, 1, 1
        )

        ######################################################### NOTE: MAMBA fusion ############################################################
        self.mamba_fusion = FusionMambareordering(self.embed_dim, depth=self.n_layers, conv_type=linear_type)
        ######################################################### NOTE: Flattening the fused feature #############################################

        self.conv_last = Conv1dSubsampler(
            in_channels=self.embed_dim,
            mid_channels=self.mid_channels,
            out_channels=self.conv_out,
        )
        
        self.linear = nn.Linear(self.embed_dim, self.fused_out_size)
        ######################################################### NOTE: Classifier ###############################################################

        self.proj_norm = LayerNorm(32)
        self.proj = nn.Linear(32, self.classifier)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm or nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    def forward(self, audio_feat, image):

        #H, W = y.shape[2:]
        ######################################################### NOTE: Image ###############################################################
        y = self.lrelu(self.low_level_feature_extraction1(image))
        y = self.lrelu(self.low_level_feature_extraction2(y))
        y = self.patch_embed(y)
        # y = self.feature_extractor_audio(image)
        # y = y.permute(0, 2, 1)
        # print(f" Image {y.shape}")
        # y = self.lrelu(self.low_level_feature_extraction1(y))
        # y = self.lrelu(self.low_level_feature_extraction2(y))
        # y = self.patch_embed(y)
        # y = self.mamba(y)
        # y = self.high_level_feature_extraction(y)
        # y = y.permute(0, 2, 1)
        # y = self.stem(y)
        # y = y.permute(0, 2, 1)
        ######################################################### NOTE: Audio ###############################################################
        # x = self.feature_extractor_audio(audio_feat)
        # x = x.permute(0, 2, 1)
        x = self.lrelu(self.low_level_feature_extraction1(audio_feat))
        x = self.lrelu(self.low_level_feature_extraction2(x))
        x = self.patch_embed(x)
        # x = x.permute(0, 2, 1)
        # x = self.patch_embed(x)
        # print(f" Audio {x.shape}")
        # x = self.mamba(x)
        # x = self.high_level_feature_extraction(x)
        # x = x.permute(0, 2, 1)
        # x = self.stem(x)
        # x = x.permute(0, 2, 1)
        ######################################################### NOTE: Fusion ###############################################################
        # z = torch.cat([x, y], dim=1)
        # z = z.permute(0, 2, 1)
        # for stage in self.stages:
        #     z = stage(z)

        output = self.mamba_fusion(y, x)
        # image_out, audio_out = self.backbone(y, x)
        output = self.conv_last(output)
        output = output.permute(1, 0, 2)
        output = self.attflat(output)
        # print(proj_feat.shape)
        # output = self.linear(output)
        ######################################################## NOTE: Flattening ############################################################
        proj_feat = self.proj_norm(output)
        ######################################################## NOTE: Classifier ############################################################
        proj_feat = self.proj(output)
        # print(z.shape)
        # z = torch.flatten(self.avgpool(self.norm(z)), 1)
        # z = F.softmax(z, dim=1)
        return proj_feat


# @register_model
# def mambaformer_s(pretrained=False, **kwargs):
#     model = MambaFormer(
#         stem_hidden_dim = 32,
#         embed_dims = [64, 128, 320, 448],
#         mlp_ratios = [8, 8, 4, 4],
#         norm_layer = partial(nn.LayerNorm, eps=1e-6),
#         depths = [3, 4, 6, 3],
#         sr_ratios = [4, 2, 1, 1],
#         **kwargs)
#     model.default_cfg = _cfg()
#     return model

# @register_model
# def mambaformer_b(pretrained=False, **kwargs):
#     model = MambaFormer(
#         stem_hidden_dim = 64,
#         embed_dims = [64, 128, 320, 512],
#         mlp_ratios = [8, 8, 4, 4],
#         norm_layer = partial(nn.LayerNorm, eps=1e-6),
#         depths = [3, 4, 12, 3],
#         sr_ratios = [4, 2, 1, 1],
#         **kwargs)
#     model.default_cfg = _cfg()
#     return model

# @register_model
# def mambaformer_l(pretrained=False, **kwargs):
#     model = MambaFormer(
#         stem_hidden_dim = 64,
#         embed_dims = [96, 192, 384, 512],
#         mlp_ratios = [8, 8, 4, 4],
#         norm_layer = partial(nn.LayerNorm, eps=1e-6),
#         depths = [3, 6, 18, 3],
#         sr_ratios = [4, 2, 1, 1],
#         **kwargs)
#     model.default_cfg = _cfg()
#     return model


# model = MambaFormer().cuda()
# x = torch.randn(2, 40000, dtype=torch.float32).cuda()
# y = torch.randn(2, 3, 150, 150, dtype=torch.float32).cuda()
# print(model(x, y).shape)
# model_parameters = filter(lambda p: p.requires_grad, model.parameters())
# params = sum([np.prod(p.size()) for p in model_parameters])
# print(params)


# %%
