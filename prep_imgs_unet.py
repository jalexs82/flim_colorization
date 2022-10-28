import argparse
import yaml
import numpy as np
import logging
import os
logging.disable(logging.WARNING)
os.environ["SM_FRAMEWORK"] = "tf.keras"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import shutil

from skimage.io import imsave
from skimage.color import gray2rgb
from keras.models import load_model

from utils import FLIMDataset, label2kcenters, LDA, hv2rgb, hsv2rgb, rgb, l_img_norm, patch_predict

valid_targs = ['lda_1ch', 'lda_2ch']
out_dim_ops = {'hv2rgb': [2], 'hsv2rgb': [3], 'gray2rgb': [1], 'rgb': [1, 2, 3], 'label': [1],
               'raw_tif': [1, 2, 3]}

parser = argparse.ArgumentParser(description='Prepare Image Data from Source and Unet Model')

parser.add_argument('-pdir', '--proj_dir', default=None, help='Project folder path for accessing images.hdf5, '
                                                              'presets.yaml, Unet model, and saving rgb images.\n'
                                                              'If none provided, will default to current folder.')
parser.add_argument('unet_name', help='Unet model name. Example ini570_outi570+p570')
parser.add_argument('-t', '--transform',
                    help=f"Transform(s) to use. For multiple transforms, separate with +.\n"
                         f"Transformations go in same order as entered.\n"
                         f"For K-Means, put number after k. Example lda_1ch+k6\n"
                         f"Max tansforms = 2\n"
                         f"Valid args: {valid_targs}")
parser.add_argument('-out', '--output', required=True, choices=list(out_dim_ops.keys()),
                    help='Output data type. Must match number of input channels after prediction\n'
                         'to output channels, except for rgb as long as not over 3 channels.')

args = parser.parse_args()

if args.proj_dir is not None:
    proj_dir = args.proj_dir
else:
    proj_dir = ''
assert os.path.exists(proj_dir), f"Project folder {proj_dir} does not exist."

src_path = f"{proj_dir}/images.hdf5"
img_grp_name = args.unet_name

if args.transform is not None:
    img_grp_name += f"_{args.transform}"
    post_transforms = args.transform.split("+")
    assert len(post_transforms) == 1, f"Too many transformations provided. Counted {len(post_transforms)}, Max: 1"
    assert all([t in valid_targs for t in post_transforms]), f"Invalid transform provided.\nMust be in {valid_targs}"
else:
    post_transforms = []

img_grp_name += f"_{args.output}"

# Create Image Group Directory
if not os.path.exists(f"{proj_dir}/rgb_images"):
    os.mkdir(f"{proj_dir}/rgb_images")

img_grp_dir = f"{proj_dir}/rgb_images/{img_grp_name}"
if os.path.exists(img_grp_dir):
    suffix = input(f"Project folder {img_grp_dir} already exists.\nPlease enter a suffix to append to folder name\n"
                   f"or enter 'q' to quit or 'w' to rewrite.")
    if suffix == 'q':
        quit()
    elif suffix == 'w':
        for fname in os.listdir(img_grp_dir):
            fpath = f"{img_grp_dir}/{fname}"
            try:
                if os.path.isfile(fpath) or os.path.islink(fpath):
                    os.unlink(fpath)
                elif os.path.isdir(fpath):
                    shutil.rmtree(fpath)
            except Exception as e:
                print(f"Could not rewrite {fpath} as {e}")
    else:
        img_grp_dir += f"_{suffix}"
        os.mkdir(img_grp_dir)
else:
    os.mkdir(img_grp_dir)
out_dir = f"{img_grp_dir}/images"
os.mkdir(out_dir)

# Load Unet Parameters
unet_dir = f"{proj_dir}/unet_models/{args.unet_name}"
with open(f"{unet_dir}/unet_params.yaml", 'r') as yml:
    unet_params = yaml.load(yml, Loader=yaml.FullLoader)

input_chs = unet_params['input_chs'].split("+")
intx_chs = [ch_name[1:] for ch_name in input_chs if 'i' in ch_name]
phx_chs = [ch_name[1:] for ch_name in input_chs if 'p' in ch_name]

output_chs = unet_params['output_chs'].split("+")
inty_chs = [ch_name[1:] for ch_name in output_chs if 'i' in ch_name]
phy_chs = [ch_name[1:] for ch_name in output_chs if 'p' in ch_name]
output_ch = list(set(inty_chs + phy_chs))[0]

# pred_ph_or_int = 'phasor' if len(phy_chs) > 0 else 'intensity'
if (len(inty_chs) > 0) and (len(phy_chs) > 0):
    pred_out_type = 'int+ph'
elif len(inty_chs) > 0:
    pred_out_type = 'int'
else:
    pred_out_type = 'ph'

# Load presets
with open(f"{proj_dir}/presets.yaml", 'r') as yml:
    presets = yaml.load(yml, Loader=yaml.FullLoader)
ph_inrange_dict = presets['input_range']
ph_invert_dict = presets['invert']

# Get background intensity threshold if appropriate
if (pred_out_type == 'ph') and not unet_params['inc_bkgd']:
    bkgd_int_thresh = presets['bkgd_int_thresh'][output_ch]
else:
    bkgd_int_thresh = 0

unet_transforms = unet_params['transform'].split("+") if unet_params['transform'] is not None else []

unet_ph_type = 'phasor'
k_centers = None
lda_type = None
unet_lda_type = None
n_kmeans = None
for t in unet_transforms:
    if 'lda' in t:
        unet_ph_type = t
        unet_lda_type = t
        lda_type = t

    elif 'k' in t:
        n_kmeans = int(t[1:])
        k_centers = np.array(presets['k_centers'][unet_ph_type][unet_params['ph_channels']][n_kmeans])

W = None
n_lda_chs = None
post_lda_type = None
for t in post_transforms:
    if 'lda' in t:
        if lda_type is not None:
            raise Exception("Cannot use LDA transform in post-process if already used in Unet")
        W = np.squeeze(np.array(presets['lda_eigv'][t][output_ch]))
        n_lda_chs = int(t[4])
        post_lda_type = t
        lda_type = t

if unet_lda_type is None and post_lda_type is None:
    ph_in_ranges = [presets['input_range']['g'][output_ch],
                    presets['input_range']['s'][output_ch]]
    ph_inverts = [presets['invert']['g'][output_ch],
                  presets['invert']['s'][output_ch]]
else:
    lda_type = unet_lda_type if unet_lda_type is not None else post_lda_type
    ph_in_ranges = [presets['input_range'][lda_type][output_ch]]
    ph_inverts = [presets['invert'][lda_type][output_ch]]

# Load model
model_path = f"{unet_dir}/model"
model = load_model(model_path, compile=False)

# Load image data and predict
flim_dsx = FLIMDataset(src_path, intx_chs, phx_chs, bkgd_int_thresh=bkgd_int_thresh, ph_inrange_dict=ph_inrange_dict)
flim_dsy = FLIMDataset(src_path, [output_ch], [], bkgd_int_thresh=bkgd_int_thresh, ph_inrange_dict=None)

img_names = flim_dsx.get_img_names()
for img_name in img_names:
    print(f"Predicting for {img_name}")
    out_path = f"{out_dir}/{img_name}.png"

    # Input
    intx_imgs = flim_dsx.get_intensity_imgs(img_name, pct_range=(0.5, 99.5), norm_range=(0, 1),
                                            get_bkgd_inds=False, dtype=np.float32)
    phx_imgs = flim_dsx.get_phasor_imgs(img_name, norm_range=(0, 1), dtype=np.float32)
    x_img = np.stack(intx_imgs + phx_imgs, axis=-1)

    # Output
    bkgd_inds = flim_dsy.get_bkgd_inds(img_name, output_ch)

    if pred_out_type == 'int':  # Regression to predict intensity
        inty_img = patch_predict(x_img, model, stepsize=0.5, pred_type='regression')
        inty_img = np.squeeze(inty_img, axis=-1)
        ph_output = None

    elif n_kmeans is None:
        pred_output = patch_predict(x_img, model, stepsize=0.5, pred_type='regression')
        if pred_out_type == 'ph':
            inty_img = flim_dsy.get_intensity_imgs(img_name, pct_range=(0.5, 99.5), norm_range=(0, 1),
                                                   get_bkgd_inds=False, dtype=np.float32)[0]
            ph_output = pred_output
        else:
            inty_img = pred_output[..., 0]
            ph_output = pred_output[..., 1:]

        if unet_ph_type == 'phasor':
            for z, ph_ch in enumerate(['g', 's']):
                ph_output[..., z] = l_img_norm(ph_output[..., z], (0, 1), ph_inrange_dict[ph_ch][output_ch])
        else:
            ph_outrange = ph_inrange_dict[unet_ph_type][output_ch]
            ph_output = l_img_norm(ph_output, (0, 1), ph_outrange)

    else:  # Segmentation to predict phasor (K-means)
        inty_img = flim_dsy.get_intensity_imgs(img_name, pct_range=(0.5, 99.5), norm_range=(0, 1),
                                               get_bkgd_inds=False, dtype=np.float32)[0]
        ph_output = patch_predict(x_img, model, stepsize=0.5, pred_type='segmentation')

        if args.output == 'label':
            out_path = f"{out_dir}/{img_name}.tif"

            out_img = ph_output.astype(np.uint8)
            out_img[bkgd_inds] = 0
            imsave(out_path, out_img)
            continue

        else:
            ph_output = label2kcenters(ph_output, k_centers, bkgd_zero=True)

    if post_lda_type is not None:
        ph_output = LDA(eig_vector=W, n_out_chs=n_lda_chs).apply_transform(ph_output)

    if args.output == 'hv2rgb':
        if ph_output.ndim == 2:
            imgs = [ph_output, inty_img]
        else:
            imgs = [ph_output[..., 0], inty_img]

        in_ranges = [ph_in_ranges[0], (0, 1)]
        inverts = [ph_inverts[0], False]
        out_img = hv2rgb(imgs, in_ranges, inverts)
        out_img[bkgd_inds] = 0

        imsave(out_path, out_img)

    elif args.output == 'hsv2rgb':
        imgs = [ph_output[..., 0], ph_output[..., 1], inty_img]
        in_ranges = [ph_in_ranges[0], ph_in_ranges[1], [0, 1]]
        # inverts = [ph_inverts[0], ph_inverts[1], False]
        inverts = [True, True, False]
        out_img = hsv2rgb(imgs, in_ranges, inverts)
        out_img[bkgd_inds] = 0

        imsave(out_path, out_img)

    elif args.output == 'gray2rgb':
        out_img = gray2rgb(l_img_norm(inty_img, [0, 1], [0, 255]))
        out_img[bkgd_inds] = 0

        imsave(out_path, out_img)

    elif args.output == 'rgb':
        imgs, in_ranges, inverts = [inty_img], [(0, 1)], [False]
        ph_imgs = [ph_output] if ph_output.ndim == 2 else [ph_output[..., i] for i in range(ph_output.shape[-1])]
        imgs.extend(ph_imgs), in_ranges.extend(ph_in_ranges), inverts.extend(ph_inverts)
        out_img = rgb(imgs, in_ranges, inverts)
        out_img[bkgd_inds] = 0

        imsave(out_path, out_img)

    elif args.output == 'raw_tif':
        out_path = f"{out_dir}/{img_name}.tif"

        imgs = [ph_output[..., 0], ph_output[..., 1], inty_img]
        out_img = np.stack(imgs, axis=0)
        out_img[bkgd_inds] = 0

        imsave(out_path, out_img)

# Save Configs
configs_path = f"{img_grp_dir}/configs.yaml"
configs = vars(args).copy()
configs['FLIMDataset_x'] = flim_dsx.__dict__
configs['FLIMDataset_y'] = flim_dsy.__dict__
with open(configs_path, 'w') as yml:
    yaml.dump(configs, yml)
