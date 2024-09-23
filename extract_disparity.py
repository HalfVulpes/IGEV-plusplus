import sys
sys.path.append('core')

import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from igev_stereo import IGEVStereo
from utils.utils import InputPadder
from PIL import Image
from matplotlib import pyplot as plt
import os
import skimage.io
import cv2
from utils.frame_utils import readPFM

DEVICE = 'cuda'

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def load_image(imfile):
    img = np.array(Image.open(imfile))
    
    # Check if the image has 3 channels (RGB). If not, convert grayscale to RGB by replicating the single channel.
    if len(img.shape) == 2:
        img = np.stack([img] * 3, axis=-1)
    
    img = img.astype(np.uint8)[..., :3]
    img = torch.from_numpy(img).permute(2, 0, 1).float()  # Permute to [C, H, W]
    return img[None].to(DEVICE)

def save_disparity_as_dspm(filename, disparity):
    """Saves the disparity map in ASCII format as a .dspm file."""
    with open(filename, 'w') as f:
        for row in disparity:
            f.write(' '.join(f'{val:.2f}' for val in row) + '\n')

def demo(args):
    model = torch.nn.DataParallel(IGEVStereo(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt))

    model = model.module
    model.to(DEVICE)
    model.eval()

    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True)

    with torch.no_grad():
        left_files = glob.glob(os.path.join(args.left_imgs, '*'))
        right_files = glob.glob(os.path.join(args.right_imgs, '*'))
        left_filenames = {os.path.basename(f): f for f in left_files}
        right_filenames = {os.path.basename(f): f for f in right_files}
        common_filenames = set(left_filenames.keys()) & set(right_filenames.keys())
        left_only = set(left_filenames.keys()) - set(right_filenames.keys())
        right_only = set(right_filenames.keys()) - set(left_filenames.keys())

        if left_only:
            print(f"Warning: The following files are only in the left directory and will be skipped:")
            for filename in left_only:
                print(f"  {filename}")
        if right_only:
            print(f"Warning: The following files are only in the right directory and will be skipped:")
            for filename in right_only:
                print(f"  {filename}")

        # Prepare sorted lists of image pairs
        sorted_common_filenames = sorted(common_filenames)
        left_images = [left_filenames[f] for f in sorted_common_filenames]
        right_images = [right_filenames[f] for f in sorted_common_filenames]

        print(f"Found {len(left_images)} image pairs. Saving files to {output_directory}/")

        for imfile1, imfile2 in tqdm(zip(left_images, right_images), total=len(left_images)):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)
            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)
            disp = model(image1, image2, iters=args.valid_iters, test_mode=True)
            disp = padder.unpad(disp)
            filename = os.path.splitext(os.path.basename(imfile1))[0]

            # Save disparity map as PNG
            png_filename = os.path.join(output_directory, f'{filename}.png')
            disp_numpy = disp.cpu().numpy().squeeze()
            plt.imsave(png_filename, disp_numpy, cmap='jet')

            # Save disparity map as numpy file
            if args.save_numpy:
                np.save(output_directory / f"{filename}.npy", disp_numpy)

            # Save disparity map as .dspm (ASCII format)
            dspm_filename = os.path.join(output_directory, f'{filename}.dspm')
            save_disparity_as_dspm(dspm_filename, disp_numpy)

            # Optional: Save disparity map as a colorized PNG using OpenCV (commented out)
            # disp_color = np.round(disp_numpy * 256).astype(np.uint16)
            # cv2.imwrite(png_filename, cv2.applyColorMap(cv2.convertScaleAbs(disp_color.squeeze(), alpha=0.01), cv2.COLORMAP_JET), [int(cv2.IMWRITE_PNG_COMPRESSION), 0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default='checkpoints/sceneflow.pth')
    parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
    parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="left/")
    parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="right/")
    parser.add_argument('--output_directory', help="directory to save output", default="output")
    parser.add_argument('--mixed_precision', action='store_true', default=True, help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=16, help='number of flow-field updates during forward pass')

    # Architecture choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--max_disp', type=int, default=768, help="max disp range")
    parser.add_argument('--s_disp_range', type=int, default=48, help="max disp of small disparity-range geometry encoding volume")
    parser.add_argument('--m_disp_range', type=int, default=96, help="max disp of medium disparity-range geometry encoding volume")
    parser.add_argument('--l_disp_range', type=int, default=192, help="max disp of large disparity-range geometry encoding volume")
    parser.add_argument('--s_disp_interval', type=int, default=1, help="disp interval of small disparity-range geometry encoding volume")
    parser.add_argument('--m_disp_interval', type=int, default=2, help="disp interval of medium disparity-range geometry encoding volume")
    parser.add_argument('--l_disp_interval', type=int, default=4, help="disp interval of large disparity-range geometry encoding volume")
    
    args = parser.parse_args()

    demo(args)