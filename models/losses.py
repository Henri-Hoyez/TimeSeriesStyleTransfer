import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["TF_USE_LEGACY_KERAS"]="1"
import tensorflow as tf
import numpy as np

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
error_classif = tf.keras.losses.SparseCategoricalCrossentropy()

def recontruction_loss(true:tf.Tensor, generated:tf.Tensor):
    diff = generated- true
    result = tf.math.reduce_mean(tf.square(diff))
    return tf.convert_to_tensor([result])

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return tf.convert_to_tensor([cross_entropy(tf.ones_like(fake_output), fake_output)])


def fixed_point_content(encoded_content_real, encoded_content_fake):
    diff = encoded_content_fake- encoded_content_real
    # return tf.reduce_mean(tf.square(diff))
    return tf.reduce_sum(tf.square(diff))

def style_classsification_loss(y_pred, y_true):
    return tf.convert_to_tensor([error_classif(y_true, y_pred)])

def local_generator_loss(crit_on_fakes:list):
    individual_losses = []

    for crit_on_fake in crit_on_fakes:
        individual_losses.append(cross_entropy(tf.ones_like(crit_on_fake), crit_on_fake))
        
    return tf.convert_to_tensor(individual_losses)

def local_discriminator_loss(crits_on_real, crits_on_fake):
    individual_losses = []

    for local_real, local_fake in zip(crits_on_real, crits_on_fake):
        l1 = cross_entropy(tf.ones_like(local_real), local_real)
        l2 = cross_entropy(tf.zeros_like(local_fake), local_fake)
        loss = l1+ l2
        individual_losses.append(loss)
        
    return individual_losses


def local_discriminator_accuracy(y_true, y_preds):
    # Y_true [BS, 1]
    #y_preds: [n_signals, BS, 1]

    accs = []
    for y_pred in y_preds:
        accs.append(tf.keras.metrics.binary_accuracy(y_true, y_pred))

    return tf.reduce_mean(accs)



def l2(x:tf.Tensor, y:tf.Tensor):
    diff = tf.square(y- x)
    _distance = tf.reduce_sum(diff, axis=-1)
    return _distance

def fixed_point_content(encoded_content_real, encoded_content_fake):
    diff = l2(encoded_content_real, encoded_content_fake)
    return tf.reduce_mean(diff)


def _pairwise_distance(a_embeddings, b_embeddings):
    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    dot_product = tf.matmul(a_embeddings, tf.transpose(b_embeddings))

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = tf.linalg.diag_part(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = tf.expand_dims(square_norm, 0) - 2.0 * dot_product + tf.expand_dims(square_norm, 1)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = tf.maximum(distances, 0.0)

    return distances

def select_index(pos_list:tf.Tensor):
    pos_indexes = tf.where(pos_list == 1)
    return tf.random.shuffle(tf.reshape(pos_indexes, (-1,)))[0]

def select_positive_negative(labels:tf.Tensor, embeddings:tf.Tensor):
    # Get a Square matrix, where there is ones when the class is the same.
    positives = tf.equal(labels, tf.transpose(labels))
    negatives = tf.logical_not(positives)

    positives = tf.cast(positives, tf.float32)
    negatives = tf.cast(negatives, tf.float32)
    
    positives = positives - tf.eye(labels.shape[0])

    pos_embs, negs_embs = [], []
    for i in range(positives.shape[0]):
        pos, neg = positives[i], negatives[i]

        pos_index = select_index(pos)
        neg_index = select_index(neg)

        pos_embs.append(embeddings[pos_index])
        negs_embs.append(embeddings[neg_index])

    return tf.convert_to_tensor(embeddings), tf.convert_to_tensor(pos_embs), tf.convert_to_tensor(negs_embs)


def get_triplet_loss(anchor_embedding, positive_embedding, negative_embedding, triplet_r=0.5):
    positive_distance= _pairwise_distance(anchor_embedding, positive_embedding)
    negative_distance= _pairwise_distance(anchor_embedding, negative_embedding)

    positive_index= tf.argmax(positive_distance, axis=1)

    pos_embedding = tf.gather(positive_embedding, positive_index)
 
    neg_indexes = tf.argmin(negative_distance, axis=1)
    
    neg_embeddings= tf.gather(negative_embedding, neg_indexes)

    positive_distances= l2(anchor_embedding, pos_embedding)
    negative_distances= l2(anchor_embedding, neg_embeddings)

    loss = tf.reduce_mean(tf.maximum(triplet_r+ positive_distances - negative_distances, 0))

    return loss

def hard_triplet(labels:tf.Tensor, embeddings:tf.Tensor, triplet_r=0.5):

    distances = _pairwise_distance(embeddings, embeddings)

    # Get a Square matrix, where there is ones when the class is the same.
    positives = tf.equal(labels, tf.transpose(labels))
    negatives = tf.logical_not(positives)

    positives = tf.cast(positives, tf.float32)
    negatives = tf.cast(negatives, tf.float32)
    
    positives = positives - tf.eye(labels.shape[0])

    positive_distances = distances* positives
    negatives_distances = distances* negatives
     
    negatives_distances = tf.where(tf.equal(negatives_distances, 0.), np.inf, negatives_distances)

    positive_index= tf.argmax(positive_distances, axis=1)
    pos_embs = tf.gather(embeddings, positive_index)

    negative_index = tf.argmin(negatives_distances, axis=1)
    neg_embs =  tf.gather(embeddings, negative_index)

    positive_distances= l2(embeddings, pos_embs)
    negative_distances= l2(embeddings, neg_embs)

    loss = tf.reduce_mean(tf.maximum(triplet_r+ positive_distances - negative_distances, 0))

    return loss


def fixed_point_disentanglement(
        es_x1_y:tf.Tensor, 
        es_x2_y:tf.Tensor, 
        es_y:tf.Tensor
        ):

    diff1 = l2(es_x1_y, es_x2_y)
    diff2 = l2(es_x1_y, es_y)

    loss = diff1- diff2
    zeros = tf.zeros_like(loss)
    loss = tf.math.maximum(loss, zeros)
    loss = tf.reduce_mean(loss)
    return loss


