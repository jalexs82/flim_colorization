import numpy as np
import os
import shutil
import logging
logging.disable(logging.WARNING)
os.environ["SM_FRAMEWORK"] = "tf.keras"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import yaml
import random
import argparse

from sklearn.utils import class_weight
from keras.optimizers import Adam
from keras.utils import Sequence, to_categorical
from keras.callbacks import ModelCheckpoint, CSVLogger

from utils import FLIMDataset
from unet_model import dense_unet

valid_targs = ['lda_1ch', 'lda_2ch'] + [f"k{n}" for n in range(1, 10)]

parser = argparse.ArgumentParser(description='Train Unet to predict FLIM from fluorescence or alternate channel FLIM.')

parser.add_argument('-pdir', '--proj_dir', default=None, help='Project folder path for accessing images.hdf5, '
                                                              'presets.yaml, and saving Unet models.\n'
                                                              'If none provided, will default to current folder.')
parser.add_argument('-in', '--input_chs', required=True, type=str,
                    help='Intensity and phasor channel(s) to train for input.\n'
                         'Put "i" before intensity channels and "p" before phasor channels.\n'
                         'Separate multiple inputs with +.\n'
                         'Example: i570+i590+p570 provides model with intensities for 570 + 590,\n'
                         ' phasor for 570')
parser.add_argument('-out', '--output_chs', required=True, type=str,
                    help='Intensity and phasor channel(s) to predict for output.\n'
                         'Put "i" before intensity channels and "p" before phasor channels.\n'
                         'Separate multiple inputs with +.\n'
                         'If intensity channel used, must be regression output (ie. no K-means).\n'
                         'Example: i570+p590 model predicts intensity for 570,\n'
                         ' phasor for 590')
parser.add_argument('-t', '--transform', default=None,
                    help=f"Transform(s) to apply to output.\n"
                         f"For multiple transforms, separate with +.\n"
                         f"Transformations go in same order as entered.\n"
                         f"If K-Means is used, output will switch from regression to classification.\n"
                         f"K-Means must come last if included\n"
                         f"For K-Means, put number after k. Example lda_1ch+k6\n"
                         f"Valid args: {valid_targs}")
parser.add_argument('-inc_b', '--inc_bkgd', action='store_true', help='Do not apply threshold to input & output.')
parser.add_argument('-prt', '--p_train', default=1, type=float, help='Proportion of training data to use. Max=1')
parser.add_argument('-ps', '--patch_size', default=256, help='Patch size of input_shape. (ex. 256 -> (256, 256, n_ch))')
parser.add_argument('-bs', '--batch_size', default=25, help='Batch size')
parser.add_argument('-nl', '--n_levels', default=5, help='Number of Unet levels (depth)')
parser.add_argument('-snf', '--start_n_filters', default=64,
                    help='Number of filters to start Unet. Increases by 2 ** layer_num for each layer')
parser.add_argument('-do', '--dropout', choices=[None, 'normal', 'spatial'], default=None, help='Dropout type')
parser.add_argument('-sdr', '--sp_dim_rate', default=0.3, help='Spatial dropout rate, only used in spatial dropout')
parser.add_argument('-lr', '--ln_rt', default=1e-5, help='Training learning rate')
parser.add_argument('-ns', '--n_samples', default=1000, help='Number of training samples per epoch')
parser.add_argument('-e', '--epochs', default=400, help='Number of training epochs')
args = parser.parse_args()

if args.proj_dir is not None:
    proj_dir = args.proj_dir
else:
    proj_dir = ''
assert os.path.exists(proj_dir), f"Project folder {proj_dir} does not exist."

src_path = f"{proj_dir}/images.hdf5"

input_chs = args.input_chs.split("+")
intx_chs = [ch_name[1:] for ch_name in input_chs if 'i' in ch_name]
phx_chs = [ch_name[1:] for ch_name in input_chs if 'p' in ch_name]
n_inputs = int(len(intx_chs) + (2 * len(phx_chs)))

output_chs = args.output_chs.split("+")
inty_chs = [ch_name[1:] for ch_name in output_chs if 'i' in ch_name]
phy_chs = [ch_name[1:] for ch_name in output_chs if 'p' in ch_name]
assert len(set(inty_chs + phy_chs)) == 1, f"Can only predict intensity and/or phasor for 1 output channel,\n" \
                                          f"provided: {set(inty_chs + phy_chs)}"
output_ch = list(set(inty_chs + phy_chs))[0]
n_outputs = int(len(inty_chs) + (2 * len(phy_chs)))

transforms = args.transform.split("+") if args.transform is not None else []

n_kmeans = None
for t in transforms:
    if 'k' in t:
        n_kmeans = int(t[1:])
        n_outputs = n_kmeans + 1

assert args.p_train <= 1, f"Invalid proportion of training provided. Must be <= 1. {args.p_train}"

# Create Unet model directory
unet_name = f"in{args.input_chs}_out{args.output_chs}"
if args.transform is not None:
    unet_name += f"_{args.transform}"
if args.p_train < 1:
    unet_name += f"_p{str(round(args.p_train, 2))[2:]}"
if args.inc_bkgd:
    unet_name += '_+bkgd'

if not os.path.exists(f"{proj_dir}/unet_models/"):
    os.mkdir(f"{proj_dir}/unet_models/")

unet_dir = f"{proj_dir}/unet_models/{unet_name}"
if os.path.exists(unet_dir):
    suffix = input(f"Unet folder {unet_dir} already exists.\nPlease enter a suffix to append to folder name\n"
                   f"or enter 'q' to quit or 'w' to rewrite.")
    if suffix == 'q':
        quit()
    elif suffix == 'w':
        for fname in os.listdir(unet_dir):
            fpath = f"{unet_dir}/{fname}"
            try:
                if os.path.isfile(fpath) or os.path.islink(fpath):
                    os.unlink(fpath)
                elif os.path.isdir(fpath):
                    shutil.rmtree(fpath)
            except Exception as e:
                print(f"Could not rewrite {fpath} as {e}")
    else:
        unet_dir += f"_{suffix}"
        os.mkdir(unet_dir)
else:
    os.mkdir(unet_dir)

train_prop = 60 / 75
val_prop = 15 / 75

input_shape = (args.patch_size, args.patch_size, n_inputs)

random.seed(3)

# Save Unet Parameters
unet_params_path = f"{unet_dir}/unet_params.yaml"
unet_params = vars(args).copy()
unet_params['input_shape'] = input_shape
unet_params['train_prop'] = train_prop
unet_params['val_prop'] = val_prop
unet_params['n_outputs'] = n_outputs
with open(unet_params_path, 'w') as yml:
    yaml.dump(unet_params, yml)


class ImgDataGen(Sequence):

    def __init__(self, x_imgs, y_imgs, n_samples, batch_size, input_shape):

        self.x_imgs = x_imgs
        self.y_imgs = y_imgs
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.input_shape = input_shape

    def __getitem__(self, index):

        batch_x, batch_y = [], []

        idxs = random.choices(range(len(self.x_imgs)), k=self.batch_size)
        for i in idxs:
            x_img, y_img = self.x_imgs[i], self.y_imgs[i]

            # Get random window
            x_start = random.randint(0, x_img.shape[1] - self.input_shape[1])
            y_start = random.randint(0, x_img.shape[0] - self.input_shape[0])
            win = np.array([y_start,
                            x_start,
                            y_start + self.input_shape[0],
                            x_start + self.input_shape[1]])
            x_win = x_img[win[0]:win[2], win[1]:win[3]]
            y_win = y_img[win[0]:win[2], win[1]:win[3]]

            # Perform augmentations
            if random.random() > 0.5:
                x_win, y_win = np.flip(x_win, axis=1), np.flip(y_win, axis=1)  # Left/Right
            if random.random() > 0.5:
                x_win, y_win = np.flip(x_win, axis=0), np.flip(y_win, axis=0)  # Up/Down
            k = random.randint(0, 2)
            x_win, y_win = np.rot90(x_win, k=k, axes=(0, 1)), np.rot90(y_win, k=k, axes=(0, 1))

            batch_x += [x_win]
            batch_y += [y_win]

        return np.asarray(batch_x), np.asarray(batch_y)

    def __len__(self):

        return self.n_samples // self.batch_size


# Load presets
with open(f"{proj_dir}/presets.yaml", 'r') as yml:
    presets = yaml.load(yml, Loader=yaml.FullLoader)
ph_inrange_dict = presets['input_range']
# Get background intensity threshold if requested
if not args.inc_bkgd:
    bkgd_int_thresh = presets['bkgd_int_thresh'][output_ch]
else:
    bkgd_int_thresh = 0

# Prepare FLIM datasets
flim_dsx = FLIMDataset(src_path, intx_chs, phx_chs, bkgd_int_thresh=bkgd_int_thresh, ph_inrange_dict=ph_inrange_dict)
flim_dsy = FLIMDataset(src_path, inty_chs, phy_chs, bkgd_int_thresh=bkgd_int_thresh, ph_inrange_dict=ph_inrange_dict)

# Initialize FLIM dataset with phasor transforms
kmeans_intype = 'phasor'
for transform in transforms:
    if 'lda' in transform:
        W = np.squeeze(np.array(presets['lda_eigv'][transform][phy_chs[0]]))
        n_lda_chs = int(transform[4])
        flim_dsy.add_ph_transform('lda', eig_vector=W, n_out_chs=n_lda_chs)
        kmeans_intype = transform

    elif 'k' in transform:
        k_centers = np.array(presets['k_centers'][kmeans_intype][phy_chs[0]][n_kmeans])
        flim_dsy.add_ph_transform('kmeans', k_centers=k_centers, return_type='label', bkgd_zero=True)

# Construct X,Y dataset
x_train, y_train = [], []
x_val, y_val = [], []
x_test, y_test = [], []

img_names = flim_dsx.get_img_names()
img_names = [name for name in img_names if name not in presets['test_names']]  # Exclude test names
if args.p_train < 1:
    img_names = random.sample(img_names, k=round(len(img_names) * args.p_train))
train_names = random.sample(img_names, k=round(len(img_names) * train_prop))
val_names = [name for name in img_names if name not in train_names]

for img_name in img_names:
    # Input
    intx_imgs = flim_dsx.get_intensity_imgs(img_name, pct_range=(0.5, 99.5), norm_range=(0, 1),
                                            get_bkgd_inds=False, dtype=np.float32)
    phx_imgs = flim_dsx.get_phasor_imgs(img_name, norm_range=(0, 1), dtype=np.float32)
    x_img = np.stack(intx_imgs + phx_imgs, axis=-1)

    # Output
    bkgd_inds = flim_dsy.get_bkgd_inds(img_name, output_ch)

    if args.transform is not None:
        klabel_img = flim_dsy.get_phasor_output(img_name)
        klabel_img[bkgd_inds] = 0
        y_img = to_categorical(klabel_img, num_classes=n_outputs)
    else:
        inty_imgs = flim_dsy.get_intensity_imgs(img_name, pct_range=(0.5, 99.5), norm_range=(0, 1),
                                                get_bkgd_inds=False, dtype=np.float32)
        for i in range(len(inty_imgs)):
            inty_imgs[i][bkgd_inds] = 0

        phy_imgs = flim_dsy.get_phasor_imgs(img_name, norm_range=(0, 1), dtype=np.float32)
        for i in range(len(phy_imgs)):
            phy_imgs[i][bkgd_inds] = 0

        y_img = np.stack(inty_imgs + phy_imgs, axis=-1)

    if img_name in train_names:
        x_train += [x_img]
        y_train += [y_img]
    elif img_name in val_names:
        x_val += [x_img]
        y_val += [y_img]

################################### RUN MODEL ####################################################
if n_kmeans is not None:  # Using k-means makes this a segmentation type problem
    from segmentation_models.losses import DiceLoss, CategoricalFocalLoss
    from segmentation_models.metrics import IOUScore, FScore

    last_activation = 'softmax'
    metrics = [IOUScore(threshold=0.5), FScore(threshold=0.5)]

    # Calculate class weights
    masks_le_1d = np.argmax(np.array(y_train), axis=-1).ravel()
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(masks_le_1d), y=masks_le_1d)
    # Make background low weight
    class_weights[0] = 0.01

    dice_loss = DiceLoss(class_weights=class_weights)
    focal_loss = CategoricalFocalLoss()
    loss = dice_loss + focal_loss

else:  # No k-means makes this a regression type problem
    last_activation = None
    metrics = ['mse', 'mae']
    loss = 'mse'


# Init data generators
train_gen = ImgDataGen(x_train, y_train, n_samples=args.n_samples, batch_size=args.batch_size, input_shape=input_shape)
val_gen = ImgDataGen(x_val, y_val, n_samples=args.n_samples, batch_size=args.batch_size, input_shape=input_shape)

model = dense_unet(input_shape, n_outputs, n_levels=args.n_levels, start_n_filters=args.start_n_filters,
                   last_activation=last_activation, dropout_type=args.dropout, sp_dim_rate=args.sp_dim_rate)

model.summary()

model.compile(optimizer=Adam(learning_rate=args.ln_rt, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.000000199),
              loss=loss, metrics=metrics)

csv_logger = CSVLogger(f"{unet_dir}/log.txt", separator=',', append=False)
model_checkpoint = ModelCheckpoint(f"{unet_dir}/model", monitor='val_loss', save_best_only=True)

model.fit(train_gen, validation_data=val_gen, epochs=args.epochs, callbacks=[model_checkpoint, csv_logger])
