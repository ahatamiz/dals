## Deep Active Lesion Segmention (DALS), Code by Ali Hatamizadeh ( http://web.cs.ucla.edu/~ahatamiz/ )

import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score
from utils import load_image,my_func,resolve_status,contoured_image
import matplotlib.pyplot as plt
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--logdir', default='network_lung', type=str)
parser.add_argument('--mu', default=0.2, type=float)
parser.add_argument('--nu', default=5.0, type=float)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--train_sum_freq', default=150, type=int)
parser.add_argument('--train_iter', default=150000, type=int)
parser.add_argument('--acm_iter_limit', default=300, type=int)
parser.add_argument('--img_resize', default=512, type=int)
parser.add_argument('--f_size', default=15, type=int)
parser.add_argument('--train_status', default=1, type=int)
parser.add_argument('--narrow_band_width', default=1, type=int)
parser.add_argument('--save_freq', default=1000, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--gpu', default='0', type=str)
args = parser.parse_args()
restore,is_training =resolve_status(args.train_status)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu


###### Demo 1 # Brain
image_add = './dataset/demo_brain/img1_input.npy'
label_add = './dataset/demo_brain/img1_label.npy'
init_seg_add = './dataset/demo_brain/img1_initseg.npy'


def re_init_phi(phi, dt):
    D_left_shift = tf.cast(tf.manip.roll(phi, -1, axis=1), dtype='float32')
    D_right_shift = tf.cast(tf.manip.roll(phi, 1, axis=1), dtype='float32')
    D_up_shift = tf.cast(tf.manip.roll(phi, -1, axis=0), dtype='float32')
    D_down_shift = tf.cast(tf.manip.roll(phi, 1, axis=0), dtype='float32')
    bp = D_left_shift - phi
    cp = phi - D_down_shift
    dp = D_up_shift - phi
    ap = phi - D_right_shift
    an = tf.identity(ap)
    bn = tf.identity(bp)
    cn = tf.identity(cp)
    dn = tf.identity(dp)
    ap = tf.clip_by_value(ap, 0, 10 ^ 38)
    bp = tf.clip_by_value(bp, 0, 10 ^ 38)
    cp = tf.clip_by_value(cp, 0, 10 ^ 38)
    dp = tf.clip_by_value(dp, 0, 10 ^ 38)
    an = tf.clip_by_value(an, -10 ^ 38, 0)
    bn = tf.clip_by_value(bn, -10 ^ 38, 0)
    cn = tf.clip_by_value(cn, -10 ^ 38, 0)
    dn = tf.clip_by_value(dn, -10 ^ 38, 0)
    area_pos = tf.where(phi > 0)
    area_neg = tf.where(phi < 0)
    pos_y = area_pos[:, 0]
    pos_x = area_pos[:, 1]
    neg_y = area_neg[:, 0]
    neg_x = area_neg[:, 1]
    tmp1 = tf.reduce_max([tf.square(tf.gather_nd(t, area_pos)) for t in [ap, bn]], axis=0)
    tmp1 += tf.reduce_max([tf.square(tf.gather_nd(t, area_pos)) for t in [cp, dn]], axis=0)
    update1 = tf.sqrt(tf.abs(tmp1)) - 1
    indices1 = tf.stack([pos_y, pos_x], 1)
    tmp2 = tf.reduce_max([tf.square(tf.gather_nd(t, area_neg)) for t in [an, bp]], axis=0)
    tmp2 += tf.reduce_max([tf.square(tf.gather_nd(t, area_neg)) for t in [cn, dp]], axis=0)
    update2 = tf.sqrt(tf.abs(tmp2)) - 1
    indices2 = tf.stack([neg_y, neg_x], 1)
    indices_final = tf.concat([indices1, indices2], 0)
    update_final = tf.concat([update1, update2], 0)
    dD = tf.scatter_nd(indices_final, update_final, shape=[input_image_size, input_image_size])
    S = tf.divide(phi, tf.square(phi) + 1)
    phi = phi - tf.multiply(dt * S, dD)

    return phi


def get_curvature(phi, x, y):
    phi_shape = tf.shape(phi)
    dim_y = phi_shape[0]
    dim_x = phi_shape[1]
    x = tf.cast(x, dtype="int32")
    y = tf.cast(y, dtype="int32")
    y_plus = tf.cast(y + 1, dtype="int32")
    y_minus = tf.cast(y - 1, dtype="int32")
    x_plus = tf.cast(x + 1, dtype="int32")
    x_minus = tf.cast(x - 1, dtype="int32")
    y_plus = tf.minimum(tf.cast(y_plus, dtype="int32"), tf.cast(dim_y - 1, dtype="int32"))
    x_plus = tf.minimum(tf.cast(x_plus, dtype="int32"), tf.cast(dim_x - 1, dtype="int32"))
    y_minus = tf.maximum(y_minus, 0)
    x_minus = tf.maximum(x_minus, 0)
    d_phi_dx = tf.gather_nd(phi, tf.stack([y, x_plus], 1)) - tf.gather_nd(phi, tf.stack([y, x_minus], 1))
    d_phi_dx_2 = tf.square(d_phi_dx)
    d_phi_dy = tf.gather_nd(phi, tf.stack([y_plus, x], 1)) - tf.gather_nd(phi, tf.stack([y_minus, x], 1))
    d_phi_dy_2 = tf.square(d_phi_dy)
    d_phi_dxx = tf.gather_nd(phi, tf.stack([y, x_plus], 1)) + tf.gather_nd(phi, tf.stack([y, x_minus], 1)) - \
                2 * tf.gather_nd(phi, tf.stack([y, x], 1))
    d_phi_dyy = tf.gather_nd(phi, tf.stack([y_plus, x], 1)) + tf.gather_nd(phi, tf.stack([y_minus, x], 1)) - \
                2 * tf.gather_nd(phi, tf.stack([y, x], 1))
    d_phi_dxy = 0.25 * (- tf.gather_nd(phi, tf.stack([y_minus, x_minus], 1)) - tf.gather_nd(phi, tf.stack(
        [y_plus, x_plus], 1)) + tf.gather_nd(phi, tf.stack([y_minus, x_plus], 1)) + tf.gather_nd(phi, tf.stack(
        [y_plus, x_minus], 1)))
    tmp_1 = tf.multiply(d_phi_dx_2, d_phi_dyy) + tf.multiply(d_phi_dy_2, d_phi_dxx) - \
            2 * tf.multiply(tf.multiply(d_phi_dx, d_phi_dy), d_phi_dxy)
    tmp_2 = tf.add(tf.pow(d_phi_dx_2 + d_phi_dy_2, 1.5), 2.220446049250313e-16)
    tmp_3 = tf.pow(d_phi_dx_2 + d_phi_dy_2, 0.5)
    tmp_4 = tf.divide(tmp_1, tmp_2)
    curvature = tf.multiply(tmp_3, tmp_4)
    mean_grad = tf.pow(d_phi_dx_2 + d_phi_dy_2, 0.5)

    return curvature, mean_grad


def get_intensity(image, masked_phi, filter_patch_size=5):
    u_1 = tf.layers.average_pooling2d(tf.multiply(image, masked_phi), [filter_patch_size, filter_patch_size], 1,padding='SAME')
    u_2 = tf.layers.average_pooling2d(masked_phi, [filter_patch_size, filter_patch_size], 1, padding='SAME')
    u_2_prime = 1 - tf.cast((u_2 > 0), dtype='float32') + tf.cast((u_2 < 0), dtype='float32')
    u_2 = u_2 + u_2_prime + 2.220446049250313e-16

    return tf.divide(u_1, u_2)


def active_contour_layer(elems):
    img = elems[0]
    init_phi = elems[1]
    map_lambda1_acl = elems[2]
    map_lambda2_acl = elems[3]
    wind_coef = 3
    zero_tensor = tf.constant(0, shape=[], dtype="int32")
    def _body(i, phi_level):
        band_index = tf.reduce_all([phi_level <= narrow_band_width, phi_level >= -narrow_band_width], axis=0)
        band = tf.where(band_index)
        band_y = band[:, 0]
        band_x = band[:, 1]
        shape_y = tf.shape(band_y)
        num_band_pixel = shape_y[0]
        window_radii_x = tf.ones(num_band_pixel) * wind_coef
        window_radii_y = tf.ones(num_band_pixel) * wind_coef

        def body_intensity(j, mean_intensities_outer, mean_intensities_inner):
            ### This can be computationally expensive. Use with fewer number of acm iterations.
            xnew = tf.cast(band_x[j], dtype="float32")
            ynew = tf.cast(band_y[j], dtype="float32")
            window_radius_x = tf.cast(window_radii_x[j], dtype="float32")
            window_radius_y = tf.cast(window_radii_y[j], dtype="float32")
            local_window_x_min = tf.cast(tf.floor(xnew - window_radius_x), dtype="int32")
            local_window_x_max = tf.cast(tf.floor(xnew + window_radius_x), dtype="int32")
            local_window_y_min = tf.cast(tf.floor(ynew - window_radius_y), dtype="int32")
            local_window_y_max = tf.cast(tf.floor(ynew + window_radius_y), dtype="int32")
            local_window_x_min = tf.maximum(zero_tensor, local_window_x_min)
            local_window_y_min = tf.maximum(zero_tensor, local_window_y_min)
            local_window_x_max = tf.minimum(tf.cast(input_image_size - 1, dtype="int32"), local_window_x_max)
            local_window_y_max = tf.minimum(tf.cast(input_image_size - 1, dtype="int32"), local_window_y_max)
            local_image = img[local_window_y_min: local_window_y_max + 1,local_window_x_min: local_window_x_max + 1]
            local_phi = phi_prime[local_window_y_min: local_window_y_max + 1,local_window_x_min: local_window_x_max + 1]
            inner = tf.where(local_phi <= 0)
            area_inner = tf.cast(tf.shape(inner)[0], dtype='float32')
            outer = tf.where(local_phi > 0)
            area_outer = tf.cast(tf.shape(outer)[0], dtype='float32')
            image_loc_inner = tf.gather_nd(local_image, inner)
            image_loc_outer = tf.gather_nd(local_image, outer)
            mean_intensity_inner = tf.cast(tf.divide(tf.reduce_sum(image_loc_inner), area_inner), dtype='float32')
            mean_intensity_outer = tf.cast(tf.divide(tf.reduce_sum(image_loc_outer), area_outer), dtype='float32')
            mean_intensities_inner = tf.concat(axis=0, values=[mean_intensities_inner[:j], [mean_intensity_inner]])
            mean_intensities_outer = tf.concat(axis=0, values=[mean_intensities_outer[:j], [mean_intensity_outer]])

            return (j + 1, mean_intensities_outer, mean_intensities_inner)

        if fast_lookup:
            phi_4d = phi_level[tf.newaxis, :, :, tf.newaxis]
            image = img[tf.newaxis, :, :, tf.newaxis]
            band_index_2 = tf.reduce_all([phi_4d <= narrow_band_width, phi_4d >= -narrow_band_width], axis=0)
            band_2 = tf.where(band_index_2)
            u_inner = get_intensity(image, tf.cast((([phi_4d <= 0])), dtype='float32')[0], filter_patch_size=f_size)
            u_outer = get_intensity(image, tf.cast((([phi_4d > 0])), dtype='float32')[0], filter_patch_size=f_size)
            mean_intensities_inner = tf.gather_nd(u_inner, band_2)
            mean_intensities_outer = tf.gather_nd(u_outer, band_2)

        else:
            mean_intensities_inner = tf.constant([0], dtype='float32')
            mean_intensities_outer = tf.constant([0], dtype='float32')
            j = tf.constant(0, dtype=tf.int32)
            _, mean_intensities_outer, mean_intensities_inner = tf.while_loop(
                lambda j, mean_intensities_outer, mean_intensities_inner:
                j < num_band_pixel, body_intensity, loop_vars=[j, mean_intensities_outer, mean_intensities_inner],
                shape_invariants=[j.get_shape(), tf.TensorShape([None]), tf.TensorShape([None])])

        lambda1 = tf.gather_nd(map_lambda1_acl, [band])
        lambda2 = tf.gather_nd(map_lambda2_acl, [band])
        curvature, mean_grad = get_curvature(phi_level, band_x, band_y)
        kappa = tf.multiply(curvature, mean_grad)
        term1 = tf.multiply(tf.cast(lambda1, dtype='float32'),tf.square(tf.gather_nd(img, [band]) - mean_intensities_inner))
        term2 = tf.multiply(tf.cast(lambda2, dtype='float32'),tf.square(tf.gather_nd(img, [band]) - mean_intensities_outer))
        force = -nu + term1 - term2
        force /= (tf.reduce_max(tf.abs(force)))
        d_phi_dt = tf.cast(force, dtype="float32") + tf.cast(mu * kappa, dtype="float32")
        dt = .45 / (tf.reduce_max(tf.abs(d_phi_dt)) + 2.220446049250313e-16)
        d_phi = dt * d_phi_dt
        update_narrow_band = d_phi
        phi_prime = phi_level + tf.scatter_nd([band], tf.cast(update_narrow_band, dtype='float32'),shape=[input_image_size, input_image_size])
        phi_prime = re_init_phi(phi_prime, 0.5)

        return (i + 1, phi_prime)

    i = tf.constant(0, dtype=tf.int32)
    phi = init_phi
    _, phi = tf.while_loop(lambda i, phi: i < iter_limit, _body, loop_vars=[i, phi])
    phi = tf.round(tf.cast((1 - tf.nn.sigmoid(phi)), dtype=tf.float32))

    return phi,init_phi, map_lambda1_acl, map_lambda2_acl

fast_lookup = True
config = tf.ConfigProto(allow_soft_placement=True)
input_shape = [args.batch_size, args.img_resize, args.img_resize, 1]
input_shape_dt = [args.batch_size, args.img_resize, args.img_resize]
iter_limit = args.acm_iter_limit
narrow_band_width = args.narrow_band_width
mu = args.mu
nu = args.nu
f_size = args.f_size
input_image_size = args.img_resize
x = tf.placeholder(shape=input_shape, dtype=tf.float32, name="x")
y = tf.placeholder(dtype=tf.float32, name="y")
out_seg = tf.placeholder(dtype=tf.float32, name="out_seg")
phase = tf.placeholder(tf.bool, name='phase')
global_step = tf.Variable(0, name='global_step', trainable=False)
map_lambda1 = tf.exp(tf.divide(tf.subtract(2.0,out_seg),tf.add(1.0,out_seg)))
map_lambda2 = tf.exp(tf.divide(tf.add(1.0, out_seg), tf.subtract(2.0, out_seg)))
y_out_dl = tf.round(out_seg)
x_acm = x[:, :, :, 0]
rounded_seg_acl = y_out_dl[:, :, :, 0]
dt_trans = tf.py_func(my_func, [rounded_seg_acl], tf.float32)
dt_trans.set_shape([args.batch_size, input_image_size, input_image_size])
phi_out,_, lambda1_tr, lambda2_tr = tf.map_fn(fn=active_contour_layer, elems=(x_acm, dt_trans, map_lambda1[:, :, :, 0], map_lambda2[:, :, :, 0]))
rounded_seg = tf.round(out_seg)
with tf.Session(config=config) as sess:

    print("########### Inference ############")

    print('Brain Demo in Progress ... ')
    image = load_image(image_add,args.batch_size,False)
    labels = load_image(label_add, args.batch_size,True)
    init_seg = np.load(init_seg_add)
    labels[labels != 0] = 1
    seg_out_acm, seg_out = sess.run([phi_out, y_out_dl],{x: image, y: labels, out_seg: init_seg, phase: False})
    seg_out = seg_out[0, :, :, 0]
    seg_out_acm = seg_out_acm[0, :, :]
    gt_mask = labels[0, :, :, 0]
    f1 = f1_score(gt_mask, seg_out, labels=None, average='micro', sample_weight=None)
    print('CNN Dice {0:0.4f}'.format(f1))
    f2 = f1_score(gt_mask, seg_out_acm, labels=None, average='micro', sample_weight=None)
    print('ACM Dice {0:0.4f}'.format(f2))
    fig = plt.figure()
    plt.subplot(1, 3, 1)
    plt.title('DALS Output, Dice:{0:0.4f}'.format(f2))
    seg_out_acm=contoured_image(seg_out_acm, image[0,:,:,0])
    plt.imshow(seg_out_acm)
    plt.subplot(1, 3, 2)
    plt.title('CNN Output, Dice:{0:0.4f}'.format(f1))
    seg_out = contoured_image(seg_out, image[0, :, :, 0])
    plt.imshow(seg_out)
    plt.subplot(1, 3, 3)
    plt.title('Radiologist Annotation')
    gt_mask = contoured_image(gt_mask, image[0, :, :, 0])
    plt.imshow(gt_mask)
    plt.show()




