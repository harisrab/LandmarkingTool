import tensorflow as tf
from deep_heatmaps_model_primary_net import DeepHeatmapsModel
import os


data_dir = '/Users/arik/Dropbox/a_mac_thesis/face_heatmap_networks/conventional_landmark_detection_dataset/'
pre_train_path = 'saved_models/0.01/model/deep_heatmaps-50000'
output_dir = os.getcwd()

flags = tf.app.flags

# mode and logging parameters
flags.DEFINE_string('mode', 'TRAIN', "'TRAIN' or 'TEST'")
flags.DEFINE_integer('print_every', 100, "print losses to screen + log every X steps")
flags.DEFINE_integer('save_every', 20000, "save model every X steps")
flags.DEFINE_integer('sample_every', 5000, "sample heatmaps + landmark predictions every X steps")
flags.DEFINE_integer('sample_grid', 9, 'number of training images in sample')
flags.DEFINE_bool('sample_to_log', True, 'samples will be saved to tensorboard log')
flags.DEFINE_integer('valid_size', 9, 'number of validation images to run')
flags.DEFINE_integer('log_valid_every', 10, 'evaluate on valid set every X epochs')
flags.DEFINE_integer('debug_data_size', 20, 'subset data size to test in debug mode')
flags.DEFINE_bool('debug', False, 'run in debug mode - use subset of the data')

# define paths
flags.DEFINE_string('output_dir', output_dir, "directory for saving models, logs and samples")
flags.DEFINE_string('save_model_path', 'model', "directory for saving the model")
flags.DEFINE_string('save_sample_path', 'sample', "directory for saving the sampled images")
flags.DEFINE_string('save_log_path', 'logs', "directory for saving the log file")
flags.DEFINE_string('img_path', data_dir, "data directory")
flags.DEFINE_string('test_model_path', 'model/deep_heatmaps-50000', "saved model to test")
flags.DEFINE_string('test_data', 'full', 'test set to use: full/common/challenging/test/art')
flags.DEFINE_string('valid_data', 'full', 'validation set to use: full/common/challenging/test/art')
flags.DEFINE_string('train_crop_dir', 'crop_gt_margin_0.25', "directory of train images cropped to bb (+margin)")
flags.DEFINE_string('img_dir_ns', 'crop_gt_margin_0.25_ns', "dir of train imgs cropped to bb + style transfer")
flags.DEFINE_string('epoch_data_dir', 'epoch_data', "directory containing pre-augmented data for each epoch")
flags.DEFINE_bool('use_epoch_data', False, "use pre-augmented data")

# pretrain parameters (for fine-tuning / resume training)
flags.DEFINE_string('pre_train_path', pre_train_path, 'pretrained model path')
flags.DEFINE_bool('load_pretrain', False, "load pretrained weight?")

# input data parameters
flags.DEFINE_integer('image_size', 256, "image size")
flags.DEFINE_integer('c_dim', 3, "color channels")
flags.DEFINE_integer('num_landmarks', 68, "number of face landmarks")
flags.DEFINE_float('sigma', 1.5, "std for heatmap generation gaussian")
flags.DEFINE_integer('scale', 1, 'scale for image normalization 255/1/0')
flags.DEFINE_float('margin', 0.25, 'margin for face crops - % of bb size')
flags.DEFINE_string('bb_type', 'gt', "bb to use -  'gt':for ground truth / 'init':for face detector output")
flags.DEFINE_bool('approx_maps', True, 'use heatmap approximation - major speed up')
flags.DEFINE_float('win_mult', 3.33335, 'gaussian filter size for approx maps: 2 * sigma * win_mult + 1')

# optimization parameters
flags.DEFINE_integer('train_iter', 100000, 'maximum training iterations')
flags.DEFINE_integer('batch_size', 10, "batch_size")
flags.DEFINE_float('learning_rate', 1e-4, "initial learning rate")
flags.DEFINE_bool('adam_optimizer', True, "use adam optimizer (if False momentum optimizer is used)")
flags.DEFINE_float('momentum', 0.95, "optimizer momentum (if adam_optimizer==False)")
flags.DEFINE_integer('step', 100000, 'step for lr decay')
flags.DEFINE_float('gamma', 0.1, 'exponential base for lr decay')
flags.DEFINE_float('reg', 1e-5, 'scalar multiplier for weight decay (0 to disable)')
flags.DEFINE_string('weight_initializer', 'xavier', 'weight initializer: random_normal / xavier')
flags.DEFINE_float('weight_initializer_std', 0.01, 'std for random_normal weight initializer')
flags.DEFINE_float('bias_initializer', 0.0, 'constant value for bias initializer')

# augmentation parameters
flags.DEFINE_bool('augment_basic', True, "use basic augmentation?")
flags.DEFINE_integer('basic_start', 0,  'min epoch to start basic augmentation')
flags.DEFINE_bool('augment_texture', False, "use artistic texture augmentation?")
flags.DEFINE_float('p_texture', 0., 'initial probability of artistic texture augmentation')
flags.DEFINE_bool('augment_geom', False, "use artistic geometric augmentation?")
flags.DEFINE_float('p_geom', 0., 'initial probability of artistic geometric augmentation')
flags.DEFINE_integer('artistic_step', -1, 'step for increasing probability of artistic augmentation in epochs')
flags.DEFINE_integer('artistic_start', 0, 'min epoch to start artistic augmentation')


FLAGS = flags.FLAGS

if not os.path.exists(FLAGS.output_dir):
    os.mkdir(FLAGS.output_dir)


def main(_):

    save_model_path = os.path.join(FLAGS.output_dir, FLAGS.save_model_path)
    save_sample_path = os.path.join(FLAGS.output_dir, FLAGS.save_sample_path)
    save_log_path = os.path.join(FLAGS.output_dir, FLAGS.save_log_path)

    # create directories if not exist
    if not os.path.exists(save_model_path):
        os.mkdir(save_model_path)
    if not os.path.exists(save_log_path):
        os.mkdir(save_log_path)
    if not os.path.exists(save_sample_path) and (not FLAGS.sample_to_log or FLAGS.mode != 'TRAIN'):
        os.mkdir(save_sample_path)

    model = DeepHeatmapsModel(
        mode=FLAGS.mode, train_iter=FLAGS.train_iter, batch_size=FLAGS.batch_size, learning_rate=FLAGS.learning_rate,
        adam_optimizer=FLAGS.adam_optimizer, momentum=FLAGS.momentum, step=FLAGS.step, gamma=FLAGS.gamma, reg=FLAGS.reg,
        weight_initializer=FLAGS.weight_initializer, weight_initializer_std=FLAGS.weight_initializer_std,
        bias_initializer=FLAGS.bias_initializer, image_size=FLAGS.image_size, c_dim=FLAGS.c_dim,
        num_landmarks=FLAGS.num_landmarks, sigma=FLAGS.sigma, scale=FLAGS.scale, margin=FLAGS.margin,
        bb_type=FLAGS.bb_type, approx_maps=FLAGS.approx_maps, win_mult=FLAGS.win_mult, augment_basic=FLAGS.augment_basic,
        basic_start=FLAGS.basic_start, augment_texture=FLAGS.augment_texture, p_texture=FLAGS.p_texture,
        augment_geom=FLAGS.augment_geom, p_geom=FLAGS.p_geom, artistic_step=FLAGS.artistic_step,
        artistic_start=FLAGS.artistic_start, output_dir=FLAGS.output_dir, save_model_path=save_model_path,
        save_sample_path=save_sample_path, save_log_path=save_log_path, test_model_path=FLAGS.test_model_path,
        pre_train_path=FLAGS.pre_train_path, load_pretrain=FLAGS.load_pretrain, img_path=FLAGS.img_path,
        test_data=FLAGS.test_data, valid_data=FLAGS.valid_data, valid_size=FLAGS.valid_size,
        log_valid_every=FLAGS.log_valid_every, train_crop_dir=FLAGS.train_crop_dir, img_dir_ns=FLAGS.img_dir_ns,
        print_every=FLAGS.print_every, save_every=FLAGS.save_every, sample_every=FLAGS.sample_every,
        sample_grid=FLAGS.sample_grid, sample_to_log=FLAGS.sample_to_log, debug_data_size=FLAGS.debug_data_size,
        debug=FLAGS.debug, use_epoch_data=FLAGS.use_epoch_data, epoch_data_dir=FLAGS.epoch_data_dir)

    if FLAGS.mode == 'TRAIN':
        model.train()
    else:
        model.eval()

if __name__ == '__main__':
    tf.app.run()
