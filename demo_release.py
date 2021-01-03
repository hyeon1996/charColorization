
import argparse
import matplotlib.pyplot as plt

from colorizers import *

parser = argparse.ArgumentParser()
parser.add_argument('-i','--img_path', type=str, default='imgs/ansel_adams3.jpg')
parser.add_argument('--use_gpu', action='store_true', help='whether to use GPU')
parser.add_argument('-o','--save_prefix', type=str, default='saved', help='will save into this file with {eccv16.png, siggraph17.png} suffixes')
opt = parser.parse_args()

# load colorizers
model = eccv16(pretrained=True).eval()
model.load_state_dict(torch.load('model/try/new_0.000130310184171107_L1_0.000130310184171107_E_479.pth',map_location='cuda:0'))
model.cuda()

# default size to process images is 256x256
# grab L channel in both original ("orig") and resized ("rs") resolutions
img = load_img(opt.img_path)
print(img.shape)
(tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256))
tens_l_rs = tens_l_rs.cuda()


print(tens_l_orig[0][0][0])


# colorizer outputs 256x256 ab map
# resize and concatenate to original L channel
img_bw = postprocess_tens(tens_l_orig, torch.cat((0*tens_l_orig,0*tens_l_orig),dim=1))
out_img = postprocess_tens(tens_l_orig, model(tens_l_rs).cpu())
plt.imsave('%s_output.png'%opt.save_prefix, out_img)
