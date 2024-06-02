import torch
from .base_model import BaseModel
from . import networks as N
import torch.nn as nn
import torch.optim as optim
from . import losses as L
import torch.nn.functional as F
import torchvision.ops as ops
from util.util import mu_tonemap
import random


# For Real-World BracketIRE+ Task
class RealPlusModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        return parser

    def __init__(self, opt):
        super(RealPlusModel, self).__init__(opt)

        self.opt = opt
        self.loss_names = ['TMRNet_l1', 'TMRNet_self', 'Total']
        self.visual_names = ['data_in', 'data_out'] 
       
        if self.isTrain:
            self.model_names = ['TMRNet', 'TMRNetPre'] 
        else:
            self.model_names = ['TMRNet']  
        
        self.optimizer_names = ['TMRNet_optimizer_%s' % opt.optimizer]

        rbsr = TMRNet(opt)
        self.netTMRNet = N.init_net(rbsr, opt.init_type, opt.init_gain, opt.gpu_ids)
        self.load_network_path(self.netTMRNet, './ckpt/syn_plus/TMRNet_model_400.pth')

        rbsr_pre = TMRNet(opt)
        self.netTMRNetPre = N.init_net(rbsr_pre, opt.init_type, opt.init_gain, opt.gpu_ids)
        N.set_requires_grad(self.netTMRNetPre.module, False)
        self.load_network_path(self.netTMRNetPre, './ckpt/syn_plus/TMRNet_model_400.pth')

        if self.isTrain:	
            self.optimizer_TMRNet = optim.AdamW(self.netTMRNet.parameters(),
                                              lr=opt.lr,
                                              betas=(opt.beta1, opt.beta2),
                                              weight_decay=opt.weight_decay)
            self.optimizers = [self.optimizer_TMRNet]

            self.criterionL1 = N.init_net(L.L1Loss(), gpu_ids=opt.gpu_ids)

    def set_input(self, input):
        self.data_raws = input['raws'].to(self.device)
        expo = input['expos'].to(self.device)
        self.image_paths = input['fname']
        self.expo = expo[:,:,None,None,None]

    def model_ema(self, decay=0.999):
        net_g_params = dict(self.netTMRNet.module.named_parameters())
        net_g_ema_params = dict(self.netTMRNetPre.module.named_parameters())
        for k in net_g_ema_params.keys():
            net_g_ema_params[k].data.mul_(decay).add_(net_g_params[k].data, alpha=1-decay)

    def forward(self):
        if self.isTrain:
            self.data_raws, = N.augment(self.data_raws,)

        self.data_raws = self.data_raws * self.expo 
        self.data_in = self.data_raws[:,0,...].squeeze(1)

        if not self.isTrain:
            B, T, C, H, W  = self.data_raws.shape
            self.data_raws = self.data_raws[:,:,:,3*H//8:5*H//8,3*W//8:5*W//8]

        if self.isTrain:
            self.data_out = self.netTMRNet(self.data_raws)
        elif not self.isTrain and not self.opt.chop:
            self.data_out = self.netTMRNet(self.data_raws)[-1]
        elif self.opt.chop:
            self.data_out = self.forward_chop(self.data_raws)

    def forward_chop(self, data_raws, chop_size=800):
        n, t, c, h, w = data_raws.shape
        s = 4
        
        num_h = h // chop_size + 1
        num_w = w // chop_size + 1
        new_h = num_h * chop_size
        new_w = num_w * chop_size
        
        pad_h = new_h - h
        pad_w = new_w - w

        pad_top = int(pad_h / 2.)
        pad_bottom = pad_h - pad_top
        pad_left = int(pad_w / 2.)
        pad_right = pad_w - pad_left

        paddings = (pad_left, pad_right, pad_top, pad_bottom)
        new_input0 = torch.nn.ReflectionPad2d(paddings)(data_raws[0])

        out = torch.zeros([1, c, new_h*s, new_w*s], dtype=torch.float32, device=data_raws.device)
        for i in range(num_h):
            for j in range(num_w):
                out[:, :, i*chop_size*s:(i+1)*chop_size*s, j*chop_size*s:(j+1)*chop_size*s] = self.netTMRNet(
                    new_input0.unsqueeze(0)[:,:,:,i*chop_size:(i+1)*chop_size, j*chop_size:(j+1)*chop_size])[-1]
        return out[:, :, pad_top*s:pad_top*s+h*s, pad_left*s:pad_left*s+w*s]
        
    def backward(self, epoch):
        with torch.no_grad():
            self.data_out_pre = self.netTMRNetPre(self.data_raws)

        self.loss_TMRNet_l1 = 1 * self.criterionL1(
            mu_tonemap(torch.clamp(self.data_out[4] / 4**2, min=0)),
            mu_tonemap(torch.clamp(self.data_out_pre[4].detach() / 4**2, 0, 1))).mean()

        t_idx = random.choices([0,1,2])[0]
        self.loss_TMRNet_self = self.opt.self_weight * self.criterionL1(
            mu_tonemap(torch.clamp(self.data_out[t_idx] / 4**2, min=0)), 
            mu_tonemap(torch.clamp(self.data_out[4].clone().detach()  / 4**2, 0, 1))).mean() 

        self.loss_Total = self.loss_TMRNet_l1 + self.loss_TMRNet_self
        self.loss_Total.backward()

    def optimize_parameters(self, epoch):
        self.forward()
        self.optimizer_TMRNet.zero_grad()
        self.backward(epoch)
        self.optimizer_TMRNet.step()
        self.model_ema(decay=0.999)


class TMRNet(nn.Module):
    def __init__(self, opt, mid_channels=64, max_residue_magnitude=10):

        super().__init__()
        self.mid_channels = mid_channels
       
        # optical flow
        self.spynet = SPyNet()
        if opt.isTrain:
            N.load_spynet(self.spynet, './ckpt/spynet/spynet_20210409-c6c1bd09.pth')

        self.dcn_alignment = DeformableAlignment(mid_channels, mid_channels, 3, padding=1, deform_groups=8,
                max_residue_magnitude=max_residue_magnitude)
        
        # feature extraction module
        self.feat_extract = ResidualBlocksWithInputConv(2*4, mid_channels, 5)

        # propagation branches
        self.backbone = nn.ModuleDict()
        self.backbone['backward'] = ResidualBlocksWithInputConv(3 * mid_channels, mid_channels, 16)

        for i in range(0,5):
            self.backbone['backward_rec_%d'%(i+1)] = ResidualBlocksWithInputConv(3 * mid_channels, mid_channels, 24) 
        self.learn_para = torch.nn.Parameter(torch.zeros(1, 5, mid_channels, 1, 1), requires_grad=True)
        
        # upsampling module
        self.reconstruction = ResidualBlocksWithInputConv(2 * mid_channels, mid_channels, 5)

        self.skipup1 = PixelShufflePack(4, mid_channels, 1, upsample_kernel=3)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.conv_hr = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)
        self.conv_last = nn.Conv2d(mid_channels, 4, 3, 1, 1)
    
        self.up_para = torch.nn.Parameter(torch.zeros(1, mid_channels, 1, 1), requires_grad=True)
        self.upsample1 = PixelShufflePack(mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample2 = PixelShufflePack(mid_channels, mid_channels, 2, upsample_kernel=3)

    def compute_flow(self, lqs):
        lqs = torch.stack((lqs[:, :, 0], lqs[:, :, 1:3].mean(dim=2), lqs[:, :, 3]), dim=2)
        lqs = torch.pow(torch.clamp(lqs, 0, 1), 1/2.2) 
        n, t, c, h, w = lqs.size()
        oth = lqs[:, 1:, :, :, :].reshape(-1, c, h, w)
        ref = lqs[:,:1, :, :, :].repeat(1,t-1,1,1,1).reshape(-1, c, h, w)
        flows_backward = self.spynet(ref, oth).view(n, t - 1, 2, h, w)
        flows_forward = flows_backward
        return flows_forward, flows_backward

    def burst_propagate(self, feats, module_name, feat_base):
        n, _, h, w = feats['spatial'][0].size()

        feat_prop = feats['spatial'][0].new_zeros(n, self.mid_channels, h, w)
        for idx in range(len(feats['spatial'])):
            feat_current = feats['spatial'][idx]
            feat = [feat_base] + [feat_current] + [feat_prop]
            feat = torch.cat(feat, dim=1)

            feat_prop = feat_prop + self.backbone[module_name](feat)

            feat_prop = feat_prop + \
                self.backbone['backward_rec_%d'%(idx+1)](
                    torch.cat([feat_base] + [feat_current] + [feat_prop], dim=1)
                    ) * self.learn_para[:,idx]

            feats[module_name].append(feat_prop)

        return feats

    def upsample(self, lqs, feats, base_feat):
        skip1 = self.skipup1(lqs[:, 0, :, :, :])
        skip1 = F.interpolate(skip1, scale_factor=4, mode='bilinear', align_corners=True)

        out = []
        t_list = [0,1,2,3,4]

        for t in t_list:
            hr = feats['backward'][t]
            hr = torch.cat([base_feat, hr], dim=1)

            hr = self.reconstruction(hr) 
            hr_up = F.interpolate(hr, scale_factor=4, mode='bilinear', align_corners=True)
        
            hr_ps = self.lrelu(self.upsample1(hr))
            hr_ps = self.lrelu(self.upsample2(hr_ps))
            
            hr_up = hr_up + skip1 + self.up_para * hr_ps
            hr_up = self.conv_last(self.lrelu(self.conv_hr(hr_up)))
            out.append(hr_up)

        return out
    
    def forward(self, lqs):
        n, t, c, h, w = lqs.size()  # (n, t, c, h, w)
        lqs_downsample = lqs.clone()

        feats = {}
        lqs_view = lqs.view(-1, c, h, w)
        lqs_in = torch.zeros([n*t, 2*c, h, w], dtype=lqs_view.dtype, device=lqs_view.device)
        lqs_in[:, 0::2, :, :] = lqs_view
        lqs_in[:, 1::2, :, :] = torch.pow(torch.clamp(lqs_view, min=0), 1/2.2) 

        feats_ = self.feat_extract(lqs_in)   # (*, C, H, W)
        h, w = feats_.shape[2:]
        feats_ = feats_.view(n, t, -1, h, w)

        _, flows_backward = self.compute_flow(lqs_downsample)
        flows_backward = flows_backward.view(-1, 2, *feats_.shape[-2:])

        ref_feat = feats_[:, :1, :, :, :].repeat(1, t-1, 1, 1, 1).view(-1, *feats_.shape[-3:])
        oth_feat = feats_[:, 1:, :, :, :].contiguous().view(-1, *feats_.shape[-3:])

        oth_feat_warped = N.flow_warp(oth_feat, flows_backward.permute(0, 2, 3, 1))
        oth_feat = self.dcn_alignment(oth_feat, ref_feat, oth_feat_warped, flows_backward)
        oth_feat = oth_feat.view(n, t-1, -1, h, w)
        ref_feat = ref_feat.view(n, t-1, -1, h, w)[:, :1, :, :, :]

        feats_ = torch.cat((ref_feat, oth_feat), dim=1)  

        feats['spatial'] = [feats_[:, i, :, :, :] for i in range(t)]
        base_feat = feats_[:, 0, :, :, :]

        # feature propagation
        module = 'backward'
        feats[module] = []
        feats = self.burst_propagate(feats, module, base_feat)

        out = self.upsample(lqs, feats, base_feat)

        return out


class ResidualBlocksWithInputConv(nn.Module):
    def __init__(self, in_channels, out_channels=64, num_blocks=30):
        super().__init__()

        main = []

        # a convolution used to match the channels of the residual blocks
        main.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True))
        main.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))

        # residual blocks
        main.append(
            N.make_layer(
                ResidualBlockNoBN, num_blocks, mid_channels=out_channels))
        
        self.main = nn.Sequential(*main)

    def forward(self, feat):
        return self.main(feat)


class ResidualBlockNoBN(nn.Module):
    def __init__(self, mid_channels=64, res_scale=1):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)

        self.relu = nn.ReLU(inplace=True)

        self.init_weights()

    def init_weights(self):
        N.init_weights(self.conv1, init_type='kaiming')
        N.init_weights(self.conv2, init_type='kaiming')
        self.conv1.weight.data *= 0.1
        self.conv2.weight.data *= 0.1

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


class PixelShufflePack(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor,
                 upsample_kernel):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.upsample_kernel = upsample_kernel
        self.upsample_conv = nn.Conv2d(
            self.in_channels,
            self.out_channels * scale_factor * scale_factor,
            self.upsample_kernel,
            padding=(self.upsample_kernel - 1) // 2)
        self.init_weights()

    def init_weights(self):
        """Initialize weights for PixelShufflePack."""
        N.init_weights(self.upsample_conv, init_type='kaiming')

    def forward(self, x):
        x = self.upsample_conv(x)
        if self.scale_factor > 1:
            x = F.pixel_shuffle(x, self.scale_factor)
        return x


class DeformableAlignment(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3, padding=1, deform_groups=8, max_residue_magnitude=10):
        super().__init__()
        
        self.max_residue_magnitude = max_residue_magnitude

        self.conv_offset = nn.Sequential(
            nn.Conv2d(2 * out_channels + 2, out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(out_channels, 27 * deform_groups, 3, 1, 1),
        )

        self.deform_conv = ops.DeformConv2d(in_channels, out_channels, kernel_size=kernel, stride=1, 
                                            padding=padding, dilation=1, bias=True, groups=deform_groups)
        
        self.init_offset()

    def init_offset(self):
        N.init_weights(self.conv_offset[-1], init_type='constant')
    
    def forward(self, cur_feat, ref_feat, warped_feat, flow):
        extra_feat = torch.cat([warped_feat, ref_feat, flow], dim=1)
        out = self.conv_offset(extra_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
        offset = offset + flow.flip(1).repeat(1, offset.size(1) // 2, 1, 1)
        mask = torch.sigmoid(mask)
        return self.deform_conv(cur_feat, offset, mask=mask)


class SPyNet(nn.Module):
    """SPyNet network structure.

    The difference to the SPyNet in [tof.py] is that
        1. more SPyNetBasicModule is used in this version, and
        2. no batch normalization is used in this version.

    Paper:
        Optical Flow Estimation using a Spatial Pyramid Network, CVPR, 2017

    Args:
        pretrained (str): path for pre-trained SPyNet. Default: None.
    """

    def __init__(self):
        super().__init__()

        self.basic_module = nn.ModuleList(
            [SPyNetBasicModule() for _ in range(6)])

        self.register_buffer(
            'mean',
            torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            'std',
            torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def compute_flow(self, ref, supp):
        """Compute flow from ref to supp.

        Note that in this function, the images are already resized to a
        multiple of 32.

        Args:
            ref (Tensor): Reference image with shape of (n, 3, h, w).
            supp (Tensor): Supporting image with shape of (n, 3, h, w).

        Returns:
            Tensor: Estimated optical flow: (n, 2, h, w).
        """
        n, _, h, w = ref.size()

        # normalize the input images
        ref = [(ref - self.mean) / self.std]
        supp = [(supp - self.mean) / self.std]

        # generate downsampled frames
        for level in range(5):
            ref.append(
                F.avg_pool2d(
                    input=ref[-1],
                    kernel_size=2,
                    stride=2,
                    count_include_pad=False))
            supp.append(
                F.avg_pool2d(
                    input=supp[-1],
                    kernel_size=2,
                    stride=2,
                    count_include_pad=False))
        ref = ref[::-1]
        supp = supp[::-1]

        # flow computation
        flow = ref[0].new_zeros(n, 2, h // 32, w // 32)
        for level in range(len(ref)):
            if level == 0:
                flow_up = flow
            else:
                flow_up = F.interpolate(
                    input=flow,
                    scale_factor=2,
                    mode='bilinear',
                    align_corners=True) * 2.0

            # add the residue to the upsampled flow
            flow = flow_up + self.basic_module[level](
                torch.cat([
                    ref[level],
                    N.flow_warp(
                        supp[level],
                        flow_up.permute(0, 2, 3, 1),
                        padding_mode='border'), flow_up
                ], 1))

        return flow

    def forward(self, ref, supp):
        """Forward function of SPyNet.

        This function computes the optical flow from ref to supp.

        Args:
            ref (Tensor): Reference image with shape of (n, 3, h, w).
            supp (Tensor): Supporting image with shape of (n, 3, h, w).

        Returns:
            Tensor: Estimated optical flow: (n, 2, h, w).
        """

        # upsize to a multiple of 32
        h, w = ref.shape[2:4]
        w_up = w if (w % 32) == 0 else 32 * (w // 32 + 1)
        h_up = h if (h % 32) == 0 else 32 * (h // 32 + 1)
        ref = F.interpolate(
            input=ref, size=(h_up, w_up), mode='bilinear', align_corners=False)
        supp = F.interpolate(
            input=supp,
            size=(h_up, w_up),
            mode='bilinear',
            align_corners=False)

        # compute flow, and resize back to the original resolution
        flow = F.interpolate(
            input=self.compute_flow(ref, supp),
            size=(h, w),
            mode='bilinear',
            align_corners=False)

        # adjust the flow values
        flow[:, 0, :, :] *= float(w) / float(w_up)
        flow[:, 1, :, :] *= float(h) / float(h_up)

        return flow


class SPyNetBasicModule(nn.Module):
    """Basic Module for SPyNet.

    Paper:
        Optical Flow Estimation using a Spatial Pyramid Network, CVPR, 2017
    """

    def __init__(self):
        super().__init__()

        self.basic_module = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3)
    )

    def forward(self, tensor_input):
        """
        Args:
            tensor_input (Tensor): Input tensor with shape (b, 8, h, w).
                8 channels contain:
                [reference image (3), neighbor image (3), initial flow (2)].

        Returns:
            Tensor: Refined flow with shape (b, 2, h, w)
        """
        return self.basic_module(tensor_input)

