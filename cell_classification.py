import numpy as np
import pandas as pd
import random
import yaml
import argparse
import logging
import os
logging.disable(logging.WARNING)
os.environ["SM_FRAMEWORK"] = "tf.keras"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from skimage.io import imread
from skimage.transform import resize
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight

from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger


parser = argparse.ArgumentParser(description='Train selected model to predict cell class labels')

parser.add_argument('rgb_proj', help='RGB project folder name.')
parser.add_argument('model_type', help='Type of classification model. Example: vgg16')
parser.add_argument('-pdir', '--proj_dir', default=None, help='Project folder path for accessing images.hdf5, '
                                                              'presets.yaml, Unet model, and saving classification model.\n'
                                                              'If none provided, will default to current folder.')
parser.add_argument('-ws', '--win_size', default=32, type=int,
                    help='Window size for area around nucleus. (ex. 256 -> (256, 256, n_ch))')
parser.add_argument('-sw', '--start_weights', default=None,
                    help='Starting weights for model. Ex. imagenet. Default None')
parser.add_argument('-bs', '--batch_size', default=50, type=int, help='Batch size')
parser.add_argument('-lr', '--ln_rt', default=2e-6, type=int, help='Training learning rate')
parser.add_argument('-ns', '--n_samples', default=1600, type=int, help='Number of training samples per epoch')
parser.add_argument('-e', '--epochs', default=150, type=int, help='Number of training epochs')

args = parser.parse_args()

keras_models = ['vgg16', 'vgg19', 'resnet50', 'resnet101', 'inception_v3', 'xception', 'densenet121', 'densenet169',
                'efficientnetb5']
assert args.model_type in keras_models

input_shape = (224, 224, 3)
random.seed(42)

if args.proj_dir is not None:
    proj_dir = args.proj_dir
else:
    proj_dir = ''
assert os.path.exists(proj_dir), f"Project folder {proj_dir} does not exist."

rgb_dir = f"{proj_dir}/rgb_images/{args.rgb_proj}/images"
model_dir_name = f"{args.model_type}"
if args.start_weights is None:
    model_dir_name += '_no_pt'

main_model_dir = f"{proj_dir}/cell_class/{model_dir_name}"
if not os.path.exists(main_model_dir):
    os.mkdir(main_model_dir)
save_dir = f"{main_model_dir}/{args.rgb_proj}"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
results_save_path = f"{save_dir}/pred_results.csv"

if args.model_type == 'resnet50':
    from tensorflow.keras.applications.resnet50 import ResNet50 as keras_model
    from tensorflow.keras.applications.resnet50 import preprocess_input
    len_base = 143
elif args.model_type == 'resnet101':
    from tensorflow.keras.applications.resnet import ResNet101 as keras_model
    from tensorflow.keras.applications.resnet import preprocess_input
elif args.model_type == 'vgg16':
    from tensorflow.keras.applications.vgg16 import VGG16 as keras_model
    from tensorflow.keras.applications.vgg16 import preprocess_input
elif args.model_type == 'vgg19':
    from tensorflow.keras.applications.vgg19 import VGG19 as keras_model
    from tensorflow.keras.applications.vgg19 import preprocess_input
elif args.model_type == 'inception_v3':
    from tensorflow.keras.applications.inception_v3 import InceptionV3 as keras_model
    from tensorflow.keras.applications.inception_v3 import preprocess_input
elif args.model_type == 'xception':
    from tensorflow.keras.applications.xception import Xception as keras_model
    from tensorflow.keras.applications.xception import preprocess_input
elif args.model_type == 'densenet121':
    from tensorflow.keras.applications.densenet import DenseNet121 as keras_model
    from tensorflow.keras.applications.densenet import preprocess_input
elif args.model_type == 'efficientnetb5':
    from tensorflow.keras.applications.efficientnet import EfficientNetB5 as keras_model
    from tensorflow.keras.applications.efficientnet import preprocess_input
else:
    raise NameError(f"{args.model_type} model type not available.")

# Save Parameters
model_params_path = f"{save_dir}/model_params.yaml"
model_params = vars(args).copy()
with open(model_params_path, 'w') as yml:
    yaml.dump(model_params, yml)


class ImgDataGen(Sequence):

    def __init__(self, x_img_prep, x_df, n_samples, batch_size, input_shape):

        self.x_img_prep = x_img_prep
        self.x_df = x_df
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.m = args.win_size // 2

    def __getitem__(self, index):

        batch_x, batch_y = [], []

        idxs = random.choices(range(self.x_df.shape[0]), k=self.batch_size)
        for i in idxs:
            image_name = self.x_df.iloc[i]['image_name']
            img_indx = x_img_idict[image_name]
            img_full = self.x_img_prep[img_indx]
            # Window center
            x = self.x_df.iloc[i]['x']
            y = self.x_df.iloc[i]['y']
            # Random shift to window
            rx = random.randint(-5, 5)
            ry = random.randint(-5, 5)

            if (x + rx + self.m > img_full.shape[1]) \
                    or (x + rx - self.m < 0) \
                    or (y + ry + self.m > img_full.shape[0]) \
                    or (y + ry - self.m < 0):
                img_full_pad = np.pad(img_full.copy(), ((100, 100), (100, 00), (0, 0)), mode='reflect')

                x_win = img_full_pad[
                        (y + ry - self.m + 100):(y + ry + self.m + 100),
                        (x + rx - self.m + 100):(x + rx + self.m + 100)
                        ]

            else:
                x_win = img_full[
                        (y + ry - self.m):(y + ry + self.m),
                        (x + rx - self.m):(x + rx + self.m)
                        ]

            # Perform augmentations
            if random.random() > 0.5:
                x_win = np.flip(x_win, axis=1)  # Left/Right
            if random.random() > 0.5:
                x_win = np.flip(x_win, axis=0)  # Up/Down
            k = random.randint(0, 2)
            x_win = np.rot90(x_win, k=k, axes=(0, 1))

            batch_x += [resize(x_win, self.input_shape)]

        y_le = self.x_df.iloc[idxs]['le'].values
        y_cat = to_categorical(y_le, n_classes)

        return np.asarray(batch_x), y_cat

    def __len__(self):

        return self.n_samples // self.batch_size


# Load images and cell class dataset
cell_df = pd.read_csv(f"{proj_dir}/cell_class.csv")
image_names = list(cell_df['image_name'].unique())
tissue_names = list(cell_df['tissue'].unique())
n_classes = len(tissue_names)

le = LabelEncoder()
le.fit(tissue_names)
cell_df['le'] = le.transform(cell_df['tissue'])
le_names = le.transform(tissue_names)

class_weights = class_weight.compute_class_weight('balanced', le_names, cell_df['le'].values)
cw_dict = {}
for le_name, weight in zip(le_names, class_weights):
    cw_dict[le_name] = weight

# Construct image dictionary
x_img_idict = {}
x_imgs = []
for i, name in enumerate(image_names):
    rgb_img = imread(f"{rgb_dir}/{name}.png")
    x_img_idict[name] = i
    x_imgs += [rgb_img]
x_img_preps = preprocess_input(np.asarray(x_imgs))

# Build model
input_t = Input(shape=input_shape)
base_model = keras_model(weights=args.start_weights, include_top=False, input_tensor=input_t)

x_model = base_model.output
x_model = GlobalAveragePooling2D()(x_model)
x_model = Dense(1024, activation='relu')(x_model)
predictions = Dense(len(tissue_names), activation='softmax')(x_model)

model = Model(inputs=input_t, outputs=predictions)
# model.summary()

model.compile(optimizer=RMSprop(learning_rate=args.ln_rt), loss='categorical_crossentropy', metrics=['accuracy'])

# Init data generators
with open(f"{proj_dir}/presets.yaml", 'r') as yml:
    presets = yaml.load(yml, Loader=yaml.FullLoader)

train_names = presets['train_names']
val_names = presets['val_names']
test_names = presets['test_names']

train_df = cell_df[cell_df['image_name'].isin(train_names)]
val_df = cell_df[cell_df['image_name'].isin(val_names)]
test_df = cell_df[cell_df['image_name'].isin(test_names)]

train_gen = ImgDataGen(x_img_preps, train_df, n_samples=args.n_samples, batch_size=args.batch_size,
                       input_shape=input_shape)
val_gen = ImgDataGen(x_img_preps, val_df, n_samples=args.n_samples, batch_size=args.batch_size,
                     input_shape=input_shape)


################################### RUN MODEL #####################################################
log_save_path = f"{save_dir}/log.txt"
model_save_path = f"{save_dir}/model"

csv_logger = CSVLogger(log_save_path, separator=',', append=False)

callbacks = [ModelCheckpoint(model_save_path, monitor='val_loss', save_best_only=True), csv_logger]

if not os.path.exists(model_save_path):
    model.fit(train_gen, validation_data=val_gen, epochs=args.epochs, class_weight=cw_dict, callbacks=callbacks)
else:
    print(f"Model folder found already. Skipping to Predictions.")

################################### RUN PREDICTIONS ###############################################
if not os.path.exists(results_save_path):
    print("Running Predictions")
    model = load_model(model_save_path, compile=False)

    m = model_params['win_size'] // 2

    results_df = pd.DataFrame()
    ap_scores, roc_aucs = [], []
    eval_types = ['train', 'val', 'test']
    for eval_type, eval_df in zip(eval_types, [train_df, val_df, test_df]):

        for image_name in eval_df['image_name'].unique():
            eval_img_df = eval_df[eval_df['image_name'] == image_name]

            img_indx = x_img_idict[image_name]
            img_full = x_img_preps[img_indx]
            full_shape = img_full.shape

            x_wins = []
            for row in eval_img_df.iterrows():
                x, y = row[1]['x'], row[1]['y']

                if (x + m > img_full.shape[1]) \
                        or (x - m < 0) \
                        or (y + m > img_full.shape[0]) \
                        or (y - m < 0):
                    img_full_pad = np.pad(img_full.copy(), ((100, 100), (100, 00), (0, 0)), mode='reflect')

                    x_win = img_full_pad[
                            (y - m + 100):(y + m + 100),
                            (x - m + 100):(x + m + 100)
                            ]

                else:
                    x_win = img_full[
                            (y - m):(y + m),
                            (x - m):(x + m)
                            ]

                x_wins.append(resize(x_win, input_shape))

            x_wins = np.array(x_wins)
            y_preds = model.predict(x_wins, batch_size=args.batch_size)
            y_pred_les = np.argmax(y_preds, axis=-1)
            y_preds_df = pd.DataFrame(y_preds, columns=le.inverse_transform(np.arange(y_preds.shape[-1])))

            df = pd.DataFrame()
            df['image_name'] = eval_img_df['image_name']
            df['eval_type'] = [eval_type for _ in range(df.shape[0])]
            df['x'] = eval_img_df['x']
            df['y'] = eval_img_df['y']
            df['gt_tissue'] = eval_img_df['tissue']
            df['pred_tissue'] = le.inverse_transform(y_pred_les)

            df = df.reset_index(drop=True)

            df = pd.concat([df, y_preds_df], axis=1)

            results_df = results_df.append(df, ignore_index=True)

    results_df.to_csv(results_save_path, index=False)

