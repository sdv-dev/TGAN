"""GAN Models."""

import tensorflow as tf
from tensorpack import StagingInput, TowerTrainer
from tensorpack.graph_builder import DataParallelBuilder, LeastLoadedDeviceSetter
from tensorpack.tfutils.tower import TowerContext, TowerFuncWrapper


class GANTrainer(TowerTrainer):
    """GanTrainer model.

    We need to set :meth:`tower_func` because it's a :class:`TowerTrainer`, and only
    :class:`TowerTrainer` supports automatic graph creation for inference during training.

    If we don't care about inference during training, using :meth:`tower_func` is not needed.
    Just calling :meth:`model.build_graph` directly is OK.

    Args:
        input_queue(tensorpack.input_source.QueueInput): Data input.
        model(tgan.GAN.GANModelDesc): Model to train.

    """

    def __init__(self, model, input_queue):
        """Initialize object."""
        super().__init__()
        inputs_desc = model.get_inputs_desc()

        # Setup input
        cbs = input_queue.setup(inputs_desc)
        self.register_callback(cbs)

        # Build the graph
        self.tower_func = TowerFuncWrapper(model.build_graph, inputs_desc)
        with TowerContext('', is_training=True):
            self.tower_func(*input_queue.get_input_tensors())

        opt = model.get_optimizer()

        # Define the training iteration by default, run one d_min after one g_min
        with tf.name_scope('optimize'):
            g_min_grad = opt.compute_gradients(model.g_loss, var_list=model.g_vars)
            g_min_grad_clip = [
                (tf.clip_by_value(grad, -5.0, 5.0), var)
                for grad, var in g_min_grad
            ]

            g_min_train_op = opt.apply_gradients(g_min_grad_clip, name='g_op')
            with tf.control_dependencies([g_min_train_op]):
                d_min_grad = opt.compute_gradients(model.d_loss, var_list=model.d_vars)
                d_min_grad_clip = [
                    (tf.clip_by_value(grad, -5.0, 5.0), var)
                    for grad, var in d_min_grad
                ]

                d_min_train_op = opt.apply_gradients(d_min_grad_clip, name='d_op')

        self.train_op = d_min_train_op


class SeparateGANTrainer(TowerTrainer):
    """A GAN trainer which runs two optimization ops with a certain ratio.

    Args:
        input(tensorpack.input_source.QueueInput): Data input.
        model(tgan.GAN.GANModelDesc): Model to train.
        d_period(int): period of each d_opt run
        g_period(int): period of each g_opt run

    """

    def __init__(self, input, model, d_period=1, g_period=1):
        """Initialize object."""
        super(SeparateGANTrainer, self).__init__()
        self._d_period = int(d_period)
        self._g_period = int(g_period)
        if not min(d_period, g_period) == 1:
            raise ValueError('The minimum between d_period and g_period must be 1.')

        # Setup input
        cbs = input.setup(model.get_inputs_desc())
        self.register_callback(cbs)

        # Build the graph
        self.tower_func = TowerFuncWrapper(model.build_graph, model.get_inputs_desc())
        with TowerContext('', is_training=True):
            self.tower_func(*input.get_input_tensors())

        opt = model.get_optimizer()
        with tf.name_scope('optimize'):
            self.d_min = opt.minimize(
                model.d_loss, var_list=model.d_vars, name='d_min')
            self.g_min = opt.minimize(
                model.g_loss, var_list=model.g_vars, name='g_min')

    def run_step(self):
        """Define the training iteration."""
        if self.global_step % (self._d_period) == 0:
            self.hooked_sess.run(self.d_min)
        if self.global_step % (self._g_period) == 0:
            self.hooked_sess.run(self.g_min)


class MultiGPUGANTrainer(TowerTrainer):
    """A replacement of GANTrainer (optimize d and g one by one) with multi-gpu support.

    Args:
        nr_gpu(int):
        input(tensorpack.input_source.QueueInput): Data input.
        model(tgan.GAN.GANModelDesc): Model to train.

    """

    def __init__(self, nr_gpu, input, model):
        """Initialize object."""
        super(MultiGPUGANTrainer, self).__init__()
        if nr_gpu <= 1:
            raise ValueError('nr_gpu must be strictly greater than 1.')

        raw_devices = ['/gpu:{}'.format(k) for k in range(nr_gpu)]

        # Setup input
        input = StagingInput(input)
        cbs = input.setup(model.get_inputs_desc())
        self.register_callback(cbs)

        # Build the graph with multi-gpu replication
        def get_cost(*inputs):
            model.build_graph(*inputs)
            return [model.d_loss, model.g_loss]

        self.tower_func = TowerFuncWrapper(get_cost, model.get_inputs_desc())
        devices = [LeastLoadedDeviceSetter(d, raw_devices) for d in raw_devices]

        cost_list = DataParallelBuilder.build_on_towers(
            list(range(nr_gpu)),
            lambda: self.tower_func(*input.get_input_tensors()),
            devices)

        # Simply average the cost here. It might be faster to average the gradients
        with tf.name_scope('optimize'):
            d_loss = tf.add_n([x[0] for x in cost_list]) * (1.0 / nr_gpu)
            g_loss = tf.add_n([x[1] for x in cost_list]) * (1.0 / nr_gpu)

            opt = model.get_optimizer()
            # run one d_min after one g_min
            g_min = opt.minimize(g_loss, var_list=model.g_vars,
                                 colocate_gradients_with_ops=True, name='g_op')

            with tf.control_dependencies([g_min]):
                d_min = opt.minimize(d_loss, var_list=model.d_vars,
                                     colocate_gradients_with_ops=True, name='d_op')

        # Define the training iteration
        self.train_op = d_min
