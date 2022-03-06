"""
https://competitions.codalab.org/competitions/17094#learn_the_details-evaluation
https://paperswithcode.com/dataset/lits17
#	Username	Score
1	DeepwiseAI	0.8220
2	liver_seg	0.7990
3	mumu_all	0.7910

Per-lesion segmentations metrics for detected lesions are the mean values for
Dice score, as well as Jaccard
volume overlap error (VOE),
relative volume difference (RVD),
average symmetric surface distance (ASSD),
maximum symmetric surface distance (MSSD).
"""

"""
volume: ?, 512, 512
segmentation: ?, 512, 512
"""

import tensorflow as tf
from os.path import basename

import numpy as np
import cv2
import utils

num_class = 3
input_shape = [160, 160, 1] # Z: 200
########################################################################################################################
def parse_fn(vol, seg): # RANK: (4, 3)
    vol = tf.transpose(vol, [2, 0, 1, 3])
    vol = utils.SWN(vol, 30, [150, 25])

    seg = tf.cast(seg, 'int32')
    seg = tf.one_hot(seg, num_class, axis=-1)
    seg = tf.transpose(seg, [2, 0, 1, 3])
    return (vol, seg)

def validation_split_fn(dataset, validation_split):
    len_dataset = tf.data.experimental.cardinality(dataset).numpy()
    valid_count = int(len_dataset * validation_split)
    print(f'[Dataset|split] Total: "{len_dataset}", Train: "{len_dataset-valid_count}", Valid: "{valid_count}"')
    return dataset.skip(valid_count), dataset.take(valid_count)

def build(batch_size, validation_split=0.1):
    assert 0 <= validation_split <= 0.5
    file_path = 'C:\\dataset\\LiTS17_160_160_200.npz'
    print(f'[Dataset] load:"{basename(file_path)}", batch size:"{batch_size}", split:"{validation_split}"')
    with np.load(file_path) as data:
        dataset = load((data['vol'], data['seg']), batch_size)
        if validation_split is not None and validation_split is not 0:
            return validation_split_fn(dataset, validation_split)
        else:
            return dataset, None

def load(data, batch_size, drop=True):
    return tf.data.Dataset.from_tensor_slices(
        data
    # ).prefetch(
    #     tf.data.experimental.AUTOTUNE
    # ).interleave(
    #     lambda x : tf.data.Dataset(x).map(parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE),
    #     cycle_length = tf.data.experimental.AUTOTUNE,
    #     num_parallel_calls = tf.data.experimental.AUTOTUNE
    # ).repeat(
    #     count=3
    # ).shuffle(
    #     4,
    #     reshuffle_each_iteration=True
    # ).cache(
    ).map(
        map_func=parse_fn,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    ).unbatch( # batch > unbatch > batch 시 cardinality = -2 로 설정됨
    ).batch(
        batch_size=batch_size,
        drop_remainder=drop,
    )


########################################################################################################################
def build_test(batch_size):
    file_path = 'C:\\dataset\\LiTS17_160_160_200_test.npz'
    with np.load(file_path) as data:
        dataset = load_test((data['vol'], data['seg']), batch_size)
    return dataset

def load_test(data, batch_size, drop=True):
    return tf.data.Dataset.from_tensor_slices(data
        ).map(map_func=parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE
        ).unbatch(  # batch > unbatch > batch 시 cardinality = -2 로 설정됨
        ).batch(batch_size=batch_size, drop_remainder=drop,
        )
########################################################################################################################
