import os
import numpy as np
import tensorflow as tf
import vgg16
import utils
import matplotlib.pyplot as plt
from skimage.transform import resize

def resize_features(features, target_size):
    resized_features = []
    for feature in features:
        resized = resize(feature[0], (target_size, target_size, feature.shape[-1]), mode='reflect', anti_aliasing=True)
        resized_features.append(resized[np.newaxis, ...])
    return np.concatenate(resized_features, axis=-1)

def visualize_and_save_combined_features(features, title, file_path):
    num_filters = features.shape[-1]
    combined_image = np.sum(features[0, :, :, :], axis=-1)
    plt.figure(figsize=(8, 8))
    plt.imshow(combined_image, cmap='viridis')
    plt.title(title)
    plt.axis('off')
    plt.savefig(file_path)
    plt.close()

img1 = utils.load_image("./test_data/tiger.jpeg")
img2 = utils.load_image("./test_data/puzzle.jpeg")

batch1 = img1.reshape((1, 224, 224, 3))
batch2 = img2.reshape((1, 224, 224, 3))

batch = np.concatenate((batch1, batch2), 0)

# 确保文件夹存在
output_dir = "Feature_image"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# TensorFlow 2.x 代码
with tf.device('/cpu:0'):
    with tf.compat.v1.Session() as sess:
        images = tf.compat.v1.placeholder(dtype=tf.float32, shape=[2, 224, 224, 3])
        feed_dict = {images: batch}

        vgg = vgg16.Vgg16()
        with tf.name_scope("content_vgg"):
            vgg.build(images)

        prob = sess.run(vgg.prob, feed_dict=feed_dict)
        print(prob)
        utils.print_prob(prob[0], './synset.txt')
        utils.print_prob(prob[1], './synset.txt')

        shallow_features = sess.run(vgg.get_shallow_features(), feed_dict=feed_dict)
        deep_features = sess.run(vgg.get_deep_features(), feed_dict=feed_dict)

        # 调整浅层特征图的尺寸
        max_shallow_size = max([f.shape[1] for f in shallow_features])
        resized_shallow_features = [resize_features([f], max_shallow_size) for f in shallow_features]
        combined_shallow_features = np.concatenate(resized_shallow_features, axis=-1)
        # 可视化并保存合并的浅层特征图
        visualize_and_save_combined_features(combined_shallow_features, 'Combined Shallow Features', os.path.join(output_dir, 'shallow_features.png'))

        # 调整深层特征图的尺寸
        max_deep_size = max([f.shape[1] for f in deep_features])
        resized_deep_features = [resize_features([f], max_deep_size) for f in deep_features]
        combined_deep_features = np.concatenate(resized_deep_features, axis=-1)
        # 可视化并保存合并的深层特征图
        visualize_and_save_combined_features(combined_deep_features, 'Combined Deep Features', os.path.join(output_dir, 'deep_features.png'))
