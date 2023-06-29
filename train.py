from Siren_LoE import Siren_LoE
from Siren import Siren
import torch
import skimage
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import time
import math

def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid

def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)


def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad

def get_cameraman_tensor(sidelength):
    img = Image.fromarray(skimage.data.camera())
    transform = Compose([
        Resize(sidelength),
        ToTensor(),
        Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
    ])
    img = transform(img)
    return img

class ImageFitting(Dataset):
    def __init__(self, sidelength):
        super().__init__()
        img = get_cameraman_tensor(sidelength)
        self.pixels = img.permute(1, 2, 0).view(-1, 1)
        self.coords = get_mgrid(sidelength, 2)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0: raise IndexError

        return self.coords, self.pixels
    
cameraman = ImageFitting(256)
dataloader = DataLoader(cameraman, batch_size=1, pin_memory=True, num_workers=0)

img_siren = Siren_LoE(in_features=2, out_features=1, hidden_features=256,
                  hidden_layers=3, outermost_linear=True)

device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )

print(torch.cuda.is_available())
print(device)
img_siren.to(device)

cameraman = ImageFitting(256)
dataloader = DataLoader(cameraman, batch_size=1, pin_memory=True, num_workers=0)

img_siren = Siren_LoE(in_features=2, out_features=1, hidden_features=256,
                  hidden_layers=3, outermost_linear=True, grid='sam')
img_siren.cuda()

total_steps = 11 # Since the whole image is our dataset, this just means 500 gradient descent steps.
steps_til_summary = 10

optim = torch.optim.Adam(lr=1e-4, params=img_siren.parameters())

model_input, ground_truth = next(iter(dataloader))
model_input, ground_truth = model_input.cuda(), ground_truth.cuda()

PSNR_list_sam = []
start = time.time()
for step in range(total_steps):
    model_output, coords = img_siren(model_input)
    loss = ((model_output - ground_truth)**2).mean()

    '''if loss != 0:
        PSNR_list_sam.append(20 * math.log10(1.0 / math.sqrt(loss)))
    else:
        PSNR_list_sam.append(float('inf'))'''

    if not step % steps_til_summary:
        print("Step %d, Total loss %0.6f" % (step, loss))
        PSNR_list_sam.append(20 * math.log10(1.0 / math.sqrt(loss)))
        #plt.imshow(model_output.cpu().view(256,256).detach().numpy())
        #plt.show()

    optim.zero_grad()
    loss.backward()
    optim.step()

end = time.time()
torch.save(img_siren.state_dict(), 'best_dncnn_model.pth')
print(end-start, "초 소요, 최종 PSNR:", PSNR_list_sam[1])