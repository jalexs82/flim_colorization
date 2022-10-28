import numpy as np
import random
import h5py
import cv2
import matplotlib.pyplot as plt

from skimage.color import gray2rgb


class FLIMDataset:
    def __init__(self, src_path, int_chs, ph_chs, bkgd_int_thresh=0, ph_in_ranges=None, ph_inverts=None,
                 ph_inrange_dict=None):
        self.src_path = src_path
        self.int_chs = int_chs  # Must be list
        self.ph_chs = ph_chs  # Must be list
        self.bkgd_int_thresh = bkgd_int_thresh  # Background threshold, all values < thresh will be 0
        self.ph_in_ranges = ph_in_ranges
        self.ph_inverts = ph_inverts
        self.ph_inrange_dict = ph_inrange_dict  # from presets. ex. {'g': {'570: [46643, 60759], ...}, 's': ...}
        self.ph_transforms = []

    def get_hv2rgb(self, img_name):
        int_imgs = self.get_intensity_imgs(img_name, pct_range=(0.5, 99.5), norm_range=(0, 255), get_bkgd_inds=False,
                                           dtype=np.uint8)
        ph_output = self.get_phasor_output(img_name)
        if ph_output.ndim == 2:
            imgs = [ph_output, int_imgs[0]]
        else:
            imgs = [ph_output[..., 0], int_imgs[0]]

        in_ranges = [self.ph_in_ranges[0], (0, 255)]
        inverts = [self.ph_inverts[0], False]

        return hv2rgb(imgs, in_ranges, inverts)

    def get_hsv2rgb(self, img_name):
        int_imgs = self.get_intensity_imgs(img_name, pct_range=(0.5, 99.5), norm_range=(0, 255), get_bkgd_inds=False,
                                           dtype=np.uint8)
        ph_output = self.get_phasor_output(img_name)
        assert ph_output.ndim == 3, f"Phasor image must be 3d (multi-channel) for hsv2rgb,\n" \
                                    f"shape: {ph_output.shape}"

        imgs = [ph_output[..., 0], ph_output[..., 1], int_imgs[0]]
        in_ranges = [self.ph_in_ranges[0], self.ph_in_ranges[1], [0, 255]]
        inverts = [self.ph_inverts[0], self.ph_inverts[1], False]

        return hsv2rgb(imgs, in_ranges, inverts)

    def get_gray2rgb(self, img_name):
        int_imgs = self.get_intensity_imgs(img_name, pct_range=(0.5, 99.5), norm_range=(0, 255), get_bkgd_inds=False,
                                           dtype=np.uint8)

        return gray2rgb(int_imgs[0])

    def get_rgb(self, img_name):
        int_imgs = self.get_intensity_imgs(img_name, pct_range=(0.5, 99.5), norm_range=(0, 255), get_bkgd_inds=False,
                                           dtype=np.uint8)
        imgs, in_ranges, inverts = int_imgs, [(0, 255) for _ in int_imgs], [False for _ in int_imgs]

        if len(self.ph_chs) > 0:
            ph_output = self.get_phasor_output(img_name)
            assert ph_output.ndim in [2, 3], f"Phasor image must be 2d or 3d (multi-channel) for rgb,\n" \
                                             f"shape: {ph_output.shape}"
            ph_imgs = [ph_output] if ph_output.ndim == 2 else [ph_output[..., i] for i in range(ph_output.shape[-1])]

            imgs.extend(ph_imgs), in_ranges.extend(self.ph_in_ranges), inverts.extend(self.ph_inverts)

        assert len(imgs) < 4, f"Too many input images provided. 3 maximum, {len(imgs)} provided."

        return rgb(imgs, in_ranges, inverts)

    def get_intensity_imgs(self, img_name, pct_range=(0.5, 99.5), norm_range=(0, 1), get_bkgd_inds=False,
                           dtype=np.float32):
        int_imgs, bkgd_inds = [], []
        for ch in self.int_chs:
            int_img = self.get_img(img_name, ch, 'int')
            if get_bkgd_inds:
                bkgd_inds.append(np.where(int_img < self.bkgd_int_thresh))
            if norm_range is not None:
                in_range = (np.percentile(int_img, pct_range[0]), np.percentile(int_img, pct_range[1]))
                int_img = l_img_norm(int_img, in_range, norm_range, dtype=dtype)
            int_imgs.append(int_img)

        if get_bkgd_inds:
            return int_imgs, bkgd_inds
        else:
            return int_imgs

    def get_phasor_imgs(self, img_name, norm_range=None, dtype=np.float32):
        ph_imgs = []
        for ch in self.ph_chs:
            for ph_ch in ['g', 's']:
                ph_img = self.get_img(img_name, ch, ph_ch)
                if norm_range is not None:
                    if self.ph_inrange_dict is not None:
                        in_range = (self.ph_inrange_dict[ph_ch][ch][0], self.ph_inrange_dict[ph_ch][ch][1])
                    else:
                        in_range = (np.min(ph_img), np.max(ph_img))
                    ph_img = l_img_norm(ph_img, in_range, norm_range, dtype=dtype)
                ph_imgs.append(ph_img)

        return ph_imgs

    def get_phasor_output(self, img_name):
        ph_imgs = []
        for ch in self.ph_chs:
            ph_imgs.append(self.get_img(img_name, ch, 'g'))
            ph_imgs.append(self.get_img(img_name, ch, 's'))
        trans_img = np.stack(ph_imgs, axis=-1)
        for transform in self.ph_transforms:
            trans_img = transform.apply_transform(trans_img)

        return trans_img

    def add_ph_transform(self, trans_type, **kwargs):
        if trans_type == 'lda':
            self.ph_transforms.append(LDA(**kwargs))
        elif trans_type == 'kmeans':
            self.ph_transforms.append(KMeans(**kwargs))

    def get_img(self, img_name, channel, flim_comp):
        with h5py.File(self.src_path, 'r') as f:

            return f[img_name][channel][flim_comp][()]

    def get_bkgd_inds(self, img_name, channel):
        int_img = self.get_img(img_name, channel, 'int')

        return np.where(int_img < self.bkgd_int_thresh)

    def get_img_names(self):
        with h5py.File(self.src_path, 'r') as f:

            return sorted([key for key in f.keys()])


class LDA:
    def __init__(self, eig_vector=None, n_out_chs=1):
        self.W = eig_vector
        self.n_out_chs = n_out_chs

    def apply_transform(self, image):
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        elif len(image.shape) != 3:
            raise Exception(f"Invalid image dimensions, counted {len(image.shape)}")

        lda_img_shape = (*image.shape[:2], self.n_out_chs)
        x_lda = [np.expand_dims(image[..., i].ravel(), axis=1) for i in range(image.shape[-1])]
        x_lda = np.hstack(x_lda)

        return x_lda.dot(self.W).reshape(lda_img_shape)


class KMeans:
    def __init__(self, k_centers=None, return_type='label', bkgd_zero=True):
        self.k_centers = k_centers
        self.return_type = return_type
        self.bkgd_zero = bkgd_zero
        self.bkgd_corr = 1 if self.bkgd_zero else 0
        self.n_centers = self.k_centers.shape[0]
        self.n_channels = self.k_centers.shape[1]

    def apply_transform(self, image):
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        elif len(image.shape) != 3:
            raise Exception(f"Invalid image dimensions, counted {len(image.shape)}")
        assert image.shape[-1] == self.n_channels,\
            f"Image channels {image.shape[-1]} do not match number of k features {self.n_channels}"

        xy_shape = (*image.shape[:2], 1)
        x_kmeans = [np.expand_dims(image[..., i].ravel(), axis=1) for i in range(image.shape[-1])]
        x_kmeans = np.hstack(x_kmeans)
        m = x_kmeans.shape[0]

        distance = np.array([]).reshape(m, 0)
        for k in range(self.n_centers):
            temp_dist = np.sum((x_kmeans - self.k_centers[k, :]) ** 2, axis=1)
            distance = np.c_[distance, temp_dist]

        klabel_img = np.argmin(distance, axis=1).reshape(xy_shape)

        if self.return_type == 'label':
            return klabel_img + self.bkgd_corr
        elif self.return_type == 'value':
            return label2kcenters(klabel_img, self.k_centers, bkgd_zero=self.bkgd_zero)
        else:
            raise Exception(f"Invalid return type, {self.return_type}")


def hsv2rgb(imgs, in_ranges, inverts):
    """ Converts input image list to rgb via hsv colorspace
    imgs: [h_img, s_img, v_img]
    in_ranges: list of each input range
    inverts: list of which channels to invert
    """
    out_ranges = [[0, 179], [0, 255], [0, 255]]

    hsv_imgs = []
    for i, img in enumerate(imgs):
        if in_ranges[i] != out_ranges[i]:
            hsv_img = l_img_norm(img, in_ranges[i], out_ranges[i], dtype=np.uint8)
        else:
            hsv_img = img.astype(np.uint8)

        if inverts[i]:
            hsv_img = cv2.bitwise_not(hsv_img) - (255 - out_ranges[i][-1])

        hsv_imgs.append(hsv_img)

    return cv2.cvtColor(np.stack(hsv_imgs, axis=-1), cv2.COLOR_HSV2RGB)


def hv2rgb(imgs, in_ranges, inverts):
    """ Converts input image list to rgb via hsv colorspace, without s channel
    imgs: [h_img, v_img]
    in_ranges: list of each input range, len == 2
    inverts: list of which channels to invert, len == 2
    """
    out_ranges = [[0, 179], [0, 255], [0, 255]]

    hv_imgs = []
    for i, img in enumerate(imgs):
        if in_ranges[i] != out_ranges[i]:
            hv_img = l_img_norm(img, in_ranges[i], out_ranges[i], dtype=np.uint8)
        else:
            hv_img = img.astype(np.uint8)

        if inverts[i]:
            hv_img = cv2.bitwise_not(hv_img) - (255 - out_ranges[i][-1])

        hv_imgs.append(hv_img)

    hv_imgs.insert(1, np.zeros(hv_imgs[0].shape, dtype=np.uint8) + 255)

    return cv2.cvtColor(np.stack(hv_imgs, axis=-1), cv2.COLOR_HSV2RGB)


def rgb(imgs, in_ranges, inverts):
    """ Converts input image list to rgb via hsv colorspace, without s channel
    imgs: list of images, len in [1, 2, 3]
    in_ranges: list of each input range, len in [1, 2, 3]
    inverts: list of which channels to invert, len in [1, 2, 3]
    """
    out_range = [0, 255]
    rgb_imgs = []
    for i, img in enumerate(imgs):
        if in_ranges[i] != out_range:
            rgb_img = l_img_norm(img, in_ranges[i], out_range, dtype=np.uint8)
        else:
            rgb_img = img.astype(np.uint8)

        if inverts[i]:
            rgb_img = cv2.bitwise_not(rgb_img)

        rgb_imgs.append(rgb_img)

    for _ in range(3 - len(rgb_imgs)):
        rgb_imgs.append(np.zeros(rgb_imgs[0].shape, dtype=np.uint8))

    return np.stack(rgb_imgs, axis=-1)


def l_img_norm(img, in_range, out_range, dtype=np.float32):
    a, b = out_range[0], out_range[1]  # lower bounds, upper bounds
    c, d = in_range[0], in_range[1]  # lower bounds, upper bounds
    norm_img = (img.copy().astype(np.float32) - c) * ((b - a) / (d - c)) + a
    norm_img[norm_img < a] = a
    norm_img[norm_img > b] = b

    return norm_img.astype(dtype)


def lda_eiv_mat(X, y):
    """ Returns eigenvector matrix for linear discriminant analysis (for 2d phasor data)
    Only works with 2 dimensions currently
    X: array of feature data (rows: datapoints, cols: dimension)
    y: label data for discrimination (eg. nuclei vs. background)
    """
    n_class = len(np.unique(y))
    n_dim = X.shape[1]

    # Mean vectors
    mean_vectors = [
        np.mean(X[y == 0], axis=0),
        np.mean(X[y == 1], axis=0)
    ]

    # Within-CLass Scatter Matrix
    S_W = np.zeros((n_dim, n_dim))
    for cl, mv in zip(range(n_class), mean_vectors):
        class_sc_mat = np.zeros((n_dim, n_dim))
        for row in X[y == cl]:
            row, mv = row.reshape(n_dim, 1), mv.reshape(n_dim, 1)
            class_sc_mat += (row - mv).dot((row - mv).T)
        S_W += class_sc_mat

    # Between-Class Scatter Matrix
    overall_mean = np.mean(X, axis=0).reshape(n_dim, 1)
    S_B = np.zeros((n_dim, n_dim))
    for i, mv in enumerate(mean_vectors):
        n = len(X[y == i])
        mv = mv.reshape(2, 1)  # make column vector
        S_B += n * (mv - overall_mean).dot((mv - overall_mean).T)

        # Solve eigenvalues/eigenvectors for S_W^-1 * S_B
    eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
    eig_vecs = [eig_vecs[:, j] for j in range(eig_vecs.shape[1])]

    # Verify that eigenvalues/eigenvectors are properly solved
    for i in range(len(eig_vals)):
        eigv = eig_vecs[i].reshape(2, 1)
        np.testing.assert_array_almost_equal(np.linalg.inv(S_W).dot(S_B).dot(eigv),
                                             eig_vals[i] * eigv,
                                             decimal=6, err_msg='', verbose=True)

    # Sort eigenvectors by eigenvalues, descending
    eig_vecs = [v for _, v in sorted(zip(list(eig_vals), eig_vecs), reverse=True)]
    eig_vals = sorted(list(eig_vals), reverse=True)

    eigv1 = eig_vecs[0]
    eigv2 = np.array([eigv1[1], -eigv1[0]])

    return np.hstack((eigv1.reshape(n_dim, 1), eigv2.reshape(n_dim, 1)))


def lda_eiv_mat2(X, y):
    """ Returns eigenvector matrix for linear discriminant analysis (for 2d phasor data)
    Only works with 2 dimensions currently
    X: array of feature data (rows: datapoints, cols: dimension)
    y: label data for discrimination (eg. nuclei vs. background)
    """
    n_class = len(np.unique(y))
    n_dim = X.shape[1]

    # Mean vectors
    mean_vectors = [np.mean(X[y == c], axis=0) for c in range(n_class)]

    # Within-CLass Scatter Matrix
    S_W = np.zeros((n_dim, n_dim))
    for cl, mv in zip(range(n_class), mean_vectors):
        class_sc_mat = np.zeros((n_dim, n_dim))
        for row in X[y == cl]:
            row, mv = row.reshape(n_dim, 1), mv.reshape(n_dim, 1)
            class_sc_mat += (row - mv).dot((row - mv).T)
        S_W += class_sc_mat

    # Between-Class Scatter Matrix
    overall_mean = np.mean(X, axis=0).reshape(n_dim, 1)
    S_B = np.zeros((n_dim, n_dim))
    for i, mv in enumerate(mean_vectors):
        n = len(X[y == i])
        mv = mv.reshape(n_dim, 1)  # make column vector
        S_B += n * (mv - overall_mean).dot((mv - overall_mean).T)

        # Solve eigenvalues/eigenvectors for S_W^-1 * S_B
    eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
    eig_vecs = [eig_vecs[:, j] for j in range(eig_vecs.shape[1])]

    # Verify that eigenvalues/eigenvectors are properly solved
    for i in range(len(eig_vals)):
        eigv = eig_vecs[i].reshape(n_dim, 1)
        np.testing.assert_array_almost_equal(np.linalg.inv(S_W).dot(S_B).dot(eigv),
                                             eig_vals[i] * eigv,
                                             decimal=6, err_msg='', verbose=True)

    # Sort eigenvectors by eigenvalues, descending
    eig_vecs = [v for _, v in sorted(zip(list(eig_vals), eig_vecs), reverse=True)]
    eig_vals = sorted(list(eig_vals), reverse=True)

    eigv1 = eig_vecs[0]
    eigv2 = np.array([eigv1[1], -eigv1[0]])

    # return np.hstack((eigv1.reshape(n_dim, 1), eigv2.reshape(n_dim, 1)))
    return eigv1.reshape(n_dim, 1)


def label2kcenters(label_img, k_centers, bkgd_zero=True):
    """ Converts from class labeled image to original k-means centers
        label_img: image with labels pertaining to k-means centers (2d)
        k_centers: predefined k-means centers (n_centers, n_channels)
        bkgd_zero: True if 0 labels pertain to background and not predefined k-means centers
    """
    n_centers = k_centers.shape[0]
    n_channels = k_centers.shape[1]
    label_min = 1 if bkgd_zero else 0

    kcenter_img = np.zeros((*label_img.shape, n_channels))
    for k_lab in range(n_centers):
        for ch in range(n_channels):
            kcenter_img[label_img == k_lab + label_min, ch] = k_centers[k_lab, ch]

    return np.squeeze(kcenter_img)


def patch_predict(x_img, model, stepsize=0.5, pred_type='segmentation'):
    """Predicts over large image in patches
    x_img: input intensity image (shape = 1024, 1024, 2)
    model: Segmentation Model
    stepsize: proportion of patch to slide prediction window for each step (ie. 0.5 = 128)
    pred_type: 'segmentation' or 'regression' type model"""
    patch_shape = model.layers[0].input_shape[0][1:]
    n_classes = model.layers[-1].output_shape[-1]
    y_img_shape = (x_img.shape[0], x_img.shape[1], n_classes)
    y_img = np.zeros(y_img_shape, dtype=np.float32)
    for i in range(0, y_img_shape[0] - int(stepsize * patch_shape[0]), int(stepsize * patch_shape[0])):
        for j in range(0, y_img_shape[1] - int(stepsize * patch_shape[1]), int(stepsize * patch_shape[1])):
            x_patch = x_img[i:i+patch_shape[0], j:j+patch_shape[1]]
            y_patch = model.predict(np.expand_dims(x_patch, axis=0))
            y_patch = np.squeeze(y_patch, axis=0)
            y_img[i:i+patch_shape[0], j:j+patch_shape[1]] += y_patch

    if pred_type == 'segmentation':
        return np.argmax(y_img, axis=-1)

    else:
        return y_img / make_div_img(y_img_shape, patch_shape, stepsize, dtype=np.float32)


def make_div_img(img_shape, patch_shape, stepsize, dtype=np.float32):
    div_img = np.zeros(img_shape, dtype=dtype)
    for i in range(0, img_shape[0] - int(stepsize * patch_shape[0]), int(stepsize * patch_shape[0])):
        for j in range(0, img_shape[1] - int(stepsize * patch_shape[1]), int(stepsize * patch_shape[1])):
            div_img[i:i + patch_shape[0], j:j + patch_shape[1]] += 1

    return div_img
