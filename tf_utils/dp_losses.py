import tensorflow as tf
from tf_utils.mac_testing import logger


def get_fc_mse_loss(tgt, ipt, w, pert, dt_bound, layer):
    """
    :param tgt: target (batch_size, n_feats)
    :param ipt: input (batch_size, n_feats)
    :param w: weight
    :param pert: additive loss perturbation for DP
    :param dt_bound: None or for dT to bound b
    :param layer: indicator for out or hidden layer loss
    :return: computed loss (scalar)
    """
    bs = ipt.get_shape()[0].value
    wx = tf.matmul(ipt, w[:-1, :]) + w[-1, :]  # (bs, dout)
    w0x = tf.stop_gradient(wx)  # since num_w_steps == 1

    assert layer in ('hid', 'out')
    f_w0x = tf.nn.relu(w0x) if layer == 'hid' else w0x
    df_w0x = tf.cast(w0x > 0, tf.float32) if layer == 'hid' else tf.ones((w0x.get_shape()))

    rec_diff = f_w0x - tgt  # (bs, dout)
    t1 = rec_diff ** 2  # (bs, dout)
    t2_factor = 2 * rec_diff * df_w0x  # (bs, dout)
    a = (tf.reduce_sum(t1) - tf.reduce_sum(t2_factor * w0x)) / (2 * bs)

    if dt_bound is not None:
        ipt_plus_bias = tf.concat([ipt[:, :, None], tf.ones((bs, 1, 1))], axis=1)
        dt_tsr = tf.matmul(ipt_plus_bias, t2_factor[:, None, :])  # (bs,din+1,1)x(bs,1,dout)->(bs,din+1,dout)
        dt_tsr_clipped = tf.clip_by_norm(dt_tsr, dt_bound, axes=[1, 2])
        b_tsr = tf.reduce_sum(dt_tsr_clipped, axis=0)  # (bs,din,dout)->(din,dout)
        b = tf.reduce_sum(b_tsr * w) / (2 * bs)
        log_dt_tsr_norms(dt_tsr)
    else:
        b = tf.reduce_sum(t2_factor * wx) / (2 * bs)  # (bs, d_out) x (bs, d_out) -> sum(bs)

    loss = a + b

    if pert is not None:
        loss += tf.reduce_sum(w * pert)  # boils down to gradient perturbation

    return loss


def get_fc_cce_loss(tgt, ipt, w, pert, dt_bound):
    """
    :param tgt: target (batch_size, n_feats)
    :param ipt: input (batch_size, n_feats)
    :param w: weight
    :param pert: additive loss perturbation for DP
    :param dt_bound:
    :return: computed loss (scalar)
    """
    bs = ipt.get_shape()[0].value
    wx = tf.matmul(ipt, w[:-1]) + w[-1]
    w0x = tf.stop_gradient(wx)  # since num_w_steps == 1

    f_w0x = tf.nn.softplus(w0x)
    df_w0x = tf.nn.sigmoid(w0x)

    df_diff = df_w0x - tgt  # (bs, dout)

    a = (tf.reduce_sum(f_w0x) - tf.reduce_sum(df_w0x * w0x)) / (2 * bs)  # linear terms in a1 and a2 cancel out

    if dt_bound is not None:
        ipt_plus_bias = tf.concat([ipt[:, :, None], tf.ones((bs, 1, 1))], axis=1)
        dt_tsr = tf.matmul(ipt_plus_bias, df_diff[:, None, :])  # (bs,din+1,1)x(bs,1,dout)->(bs,din+1,dout)
        dt_tsr_clipped = tf.clip_by_norm(dt_tsr, dt_bound, axes=[1, 2])
        b_tsr = tf.reduce_sum(dt_tsr_clipped, axis=0)  # (bs,din+1,dout)->(din+1,dout)
        b = tf.reduce_sum(b_tsr * w) / (2 * bs)  # compute b with w
    else:  # clipping sensitivity sigma
        b = tf.reduce_sum(df_diff * wx) / (2 * bs)

    loss = a + b

    if pert is not None:
        loss += tf.reduce_sum(w * pert)  # boils down to gradient perturbation

    return loss


def log_dt_tsr_norms(dt_tsr):
    dt_norms = tf.norm(dt_tsr, ord='fro', axis=[1, 2])
    median_norm = tf.contrib.distributions.percentile(dt_norms, q=50.)
    is_first = 'hist_dt_norms' not in logger.ops.keys()
    layer = 0 if is_first else len(logger.ops['hist_dt_norms'])
    hist_op = tf.summary.histogram('dt_norms/layer_{}'.format(layer), dt_norms)
    median_op = tf.summary.scalar('median_dt_norms/layer_{}'.format(layer), median_norm)
    logger.ops['hist_dt_norms'] = [hist_op] if is_first else logger.ops['hist_dt_norms'] + [hist_op]
    logger.ops['median_dt_norms'] = [median_op] if is_first else logger.ops['median_dt_norms'] + [median_op]
