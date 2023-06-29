from torch import nn
import torch
import numpy as np
import skimage
import matplotlib.pyplot as plt

import math

#대조군 슈퍼픽셀 알고리즘 적용
from sklearn.preprocessing import normalize
def seg_superpixel(layer_num, method="slic"):
    img = skimage.util.img_as_float(skimage.data.camera()[::2, ::2])
    if method == "slic":
      segments_superpixel = skimage.segmentation.slic(img, n_segments=4**(layer_num+1), compactness=0.01, sigma=1, start_label=1, channel_axis=None)
    elif method == "felzen":
      segments_superpixel = skimage.segmentation.felzenszwalb(img, scale=100, sigma=0.5, min_size=4**(layer_num+1))
    elif method == "watershed":
      gradient = skimage.filters.sobel(img)
      segments_superpixel = skimage.segmentation.watershed(gradient, markers=4**(layer_num+1), compactness=0.001)
    elif method == "quick":
      img = cv2.cvtColor((img*255).astype("uint8"),cv2.COLOR_GRAY2RGB)
      segments_superpixel = skimage.segmentation.quickshift(img, kernel_size=3, max_dist=6, ratio=0.25*layer_num)
    plt.imshow(skimage.segmentation.mark_boundaries(img, segments_superpixel))
    plt.show()
    return segments_superpixel

#Segment Anything model 마스크 구현
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2
#생성된 마스크 표시
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

#레이어용 마스크 생성 (4색으로 표시되는 거)
def seg_sam(layer_num):
    img = skimage.util.img_as_float(skimage.data.camera()[::2, ::2])*255 #ground truth 불러옴
    img = cv2.cvtColor(img.astype("uint8"),cv2.COLOR_GRAY2RGB)

    sam = sam_model_registry["vit_h"](checkpoint="model/sam_vit_h_4b8939.pth")
    sam.to(device="cuda")

    mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=4**(layer_num),
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=128,  # Requires open-cv to run post-processing
)
    segment = np.zeros_like(img)
    id = layer_num
    masks = mask_generator.generate(img)
    sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True) #reverse=True면 마스크 id가 넓은 곳->좁은 곳 순으로 할당
    for i in sorted_masks:
        for m in range(256):
            for n in range(256):
                if i['segmentation'][m][n]:
                    segment[m][n] = id % 4
        id += 1
    plt.imshow(img)
    show_anns(masks)
    plt.axis('off')
    plt.show()
    return cv2.cvtColor(segment, cv2.COLOR_RGB2GRAY)

#Levels-of_Experts 적용
class SineLayer_LoE(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30, layer_num=0, N=4, grid='fine_to_coarse', coord_batch_size=256*256, is_last=False):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.is_last = is_last

        self.in_features = in_features
        self.out_features = out_features
        self.N = N
        self.layer_num = layer_num
        self.linear = nn.ModuleList([nn.Linear(in_features, out_features, bias=bias) for i in range(self.N)])

        self.init_weights()

        self.grid = grid
        self.coord_B = coord_batch_size
        self.tile_id = None

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                for i in range(self.N):
                    self.linear[i].weight.uniform_(-1 / self.in_features,
                                                    1 / self.in_features)
            else:
                for i in range(self.N):
                    self.linear[i].weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                                    np.sqrt(6 / self.in_features) / self.omega_0)

    #LoE 가중치 id 
    def get_affine_transform(self, in_coords, layer_num=1):
        if self.tile_id is None:
            H = int(math.sqrt(self.N))
            _A = torch.tensor([])
            _b = None

            if self.grid == 'slic' or self.grid == 'watershed' or self.grid == 'felzen' or self.grid == 'quick':
                if layer_num == 4:
                    _A = 1
                    _b = 0.0
                else:
                    self.tile_id = seg_superpixel(layer_num, method=self.grid)
                    temp=(torch.tensor(self.tile_id).view(self.coord_B, 1).long()-1) // (H**2)
                    self.tile_id = torch.tensor(self.tile_id).view(self.coord_B, 1).long() % (H**2)
                    draw_tile = torch.tensor(self.tile_id).view(256,256).detach().numpy() * 85
                    plt.imshow(draw_tile)
                    title = self.grid + " Layer " + str(layer_num)
                    plt.title(title)
                    plt.show()
                    return self.tile_id

            elif self.grid == 'sam':
                if layer_num == 4:
                    _A = 1
                    _b = 0.0
                else:
                    self.tile_id = seg_sam(layer_num)
                    self.tile_id = torch.tensor(self.tile_id).view(self.coord_B, 1).long() % (H**2)
                    draw_tile = torch.tensor(self.tile_id).view(256,256).detach().numpy() * 85
                    plt.imshow(draw_tile)
                    title = self.grid + " Layer " + str(layer_num)
                    plt.title(title)
                    plt.show()
                    return self.tile_id

            elif self.grid == 'gray_code':
                if layer_num == 1:
                    _A = 2
                    _b = 0.0
                elif layer_num > 1:
                    _A = 2 ** (layer_num-1)
                    _b = 0.5

            elif self.grid == 'quad_tree':
                _A = 2 ** (layer_num)
                _b = 0.0

            elif self.grid == 'fine_to_coarse':
                _A = 2 ** (5-layer_num)
                _b = 0.0

            affine_feats = in_coords*_A + _b
            x, y = affine_feats[0, :, 0], affine_feats[0, :, 1]
            x = torch.floor(x).long() % H
            y = torch.floor(y).long() % H
            self.tile_id = (H*x + y).view(self.coord_B, 1)
            draw_tile = self.tile_id.cpu().view(256,256).detach().numpy() * 85
            plt.imshow(draw_tile)
            title = self.grid + " Layer " + str(layer_num)
            plt.title(title)
            plt.show()
        return self.tile_id

    def positional_dependent_linear_1d(self, in_feats, in_coords):
        """Linear layer with position-dependent weight.
        Assuming the input coordinate is 1D.
        Args:
            weight (N * Cout * Cin tensor): Tile of N weight matrices
            bias (Cout tensor): Bias vector
            in_feats (B * Cin tensor): Batched input features
            in_coords (B * 2 tensor): Batched input coordinates
            alpha (scalar): Scale of input coordinates
            beta (scalar): Translation of input coordinates
        Returns:
            out_feats (B * Cout tensor): Batched output features
        """
        B = in_feats.size(1) # Batch size
        N = self.N # Tile size
        Cout = self.out_features # Out channel count
        Cin = self.in_features # In channel count
        # In the actual implementation, the following lines are fused into a CUDA kernel.
        tile_id = self.get_affine_transform(in_coords, layer_num=self.layer_num)

        out_feats = torch.empty([65536, Cout]).cuda()
        for t in range(N):
            mask = tile_id == t
            mask = mask.cuda()
            sel_in_feats = torch.masked_select(in_feats, mask).reshape(-1, Cin)
            sel_weight = self.linear[t].weight.cuda()
            sel_out_feats = sel_in_feats @ sel_weight.T
            out_feats.masked_scatter_(mask, sel_out_feats)

        return out_feats

    def forward(self, input):
        in_feats, in_coords = input
        output = torch.sin(self.omega_0 * self.positional_dependent_linear_1d(in_feats, in_coords).view(self.coord_B, -1))
        if self.is_last:
            return output
        else:
            return output, in_coords

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        in_feats, in_coords = input
        intermediate = self.omega_0 * self.positional_dependent_linear_1d(in_feats, in_coords).view(self.coord_B, -1)
        if self.is_last:
            return torch.sin(intermediate), intermediate
        else:
            return torch.sin(intermediate), in_coords, intermediate


class Siren_LoE(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False,
                 first_omega_0=30, hidden_omega_0=30., grid='fine_to_coarse'):
        super().__init__()
        self.grid = grid
        self.net = []
        self.net.append(SineLayer_LoE(in_features, hidden_features,
                                  is_first=True, omega_0=first_omega_0, layer_num=1, grid=grid))

        for i in range(hidden_layers-1):
            self.net.append(SineLayer_LoE(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0, layer_num=i+2, grid=grid))

        if outermost_linear:
            self.net.append(SineLayer_LoE(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0, layer_num=i+3, grid=grid, is_last=True))
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer_LoE(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0, layer_num=i+2, grid=grid))
            self.net.append(SineLayer_LoE(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0, layer_num=i+3, grid=grid, is_last=True))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net((coords, coords))
        return output, coords

    def forward_with_activations(self, coords, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!'''
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer_LoE):
                x, intermed = layer.forward_with_intermediate(x)

                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()

                activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else:
                x = layer(x)

                if retain_grad:
                    x.retain_grad()

            activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations