"""One CIFAR-10 dataset and CNN definition for the five native backends.

All models consume NCHW float32 images and share OIHW convolution weights,
input-by-output dense weights, cross entropy and an explicit SGD rate.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np


BACKENDS = ("torch", "tf", "jax", "paddle", "tinygrad")
ARCHITECTURE = "Conv(3,8,3,pad=1)-ReLU-AvgPool(2)-Conv(8,16,3,pad=1)-ReLU-AvgPool(2)-Flatten(NCHW)-Dense(1024,32)-ReLU-Dense(32,10)"


def arrays_hash(values):
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        value = np.asarray(value)
        digest.update(name.encode())
        digest.update(str(value.shape).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def prepare(output, data_root, *, seed=20261006, train_samples=5000,
            test_samples=1000, clients=3, batch_size=50, rounds=10, lr=.05):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    source = Path(data_root) / "cifar-10-batches-py"

    def read(files):
        images, labels = [], []
        for path in files:
            with path.open("rb") as stream:
                batch = pickle.load(stream, encoding="bytes")
            images.append(batch[b"data"].reshape(-1, 3, 32, 32))
            labels.extend(batch[b"labels"])
        return np.concatenate(images), np.asarray(labels, dtype=np.int64)

    train_x, train_y = read([source / f"data_batch_{i}" for i in range(1, 6)])
    test_x, test_y = read([source / "test_batch"])
    rng = np.random.default_rng(seed)

    def subset(images, labels, count):
        if count % 10 or count > len(labels):
            raise ValueError("Subset size must be a multiple of ten within the official split")
        positions = np.concatenate([rng.permutation(np.flatnonzero(labels == label))[:count // 10]
                                    for label in range(10)])
        rng.shuffle(positions)
        return images[positions], labels[positions], positions

    train_x, train_y, train_indices = subset(train_x, train_y, train_samples)
    test_x, test_y, test_indices = subset(test_x, test_y, test_samples)
    # Fixed-size captures use complete batches on every physical client. Assign
    # whole batches rather than discarding or padding real training examples.
    if train_samples % batch_size or test_samples % batch_size:
        raise ValueError("Sample sizes must be divisible by batch size")
    batches = np.array_split(np.arange(train_samples // batch_size), clients)
    partitions = [np.concatenate([np.arange(i * batch_size, (i + 1) * batch_size)
                                  for i in indices]) for indices in batches]
    assert sorted(np.concatenate(partitions).tolist()) == list(range(train_samples))
    sizes = []
    for index, positions in enumerate(partitions):
        np.savez_compressed(output / f"client_{index}.npz", images=train_x[positions],
                            labels=train_y[positions], source_indices=train_indices[positions])
        sizes.append(len(positions))
    np.savez_compressed(output / "evaluation.npz", images=test_x, labels=test_y,
                        source_indices=test_indices)
    np.savez_compressed(output / "sample.npz", images=train_x[:batch_size], labels=train_y[:batch_size])
    shapes = {"conv1": (8, 3, 3, 3), "conv2": (16, 8, 3, 3),
              "fc1": (1024, 32), "fc2": (32, 10)}
    weights = {}
    for name, shape in shapes.items():
        fan_in = int(np.prod(shape[1:])) if name.startswith("conv") else shape[0]
        weights[name + "_w"] = (rng.standard_normal(shape) * np.sqrt(2 / fan_in)).astype(np.float32)
        weights[name + "_b"] = np.zeros(shape[0] if name.startswith("conv") else shape[1], np.float32)
    np.savez(output / "initial_weights.npz", **weights)
    config = {"schema": "splitfleet.physical-multibackend-cifar.v1", "source": "real",
              "dataset": "CIFAR-10 official train/test splits", "architecture": ARCHITECTURE,
              "seed": seed, "train_samples": train_samples, "test_samples": test_samples,
              "clients": clients, "partition_sizes": sizes, "batch_size": batch_size,
              "rounds": rounds, "local_epochs": 1, "learning_rate": lr,
              "optimizer": "SGD without momentum, reset each round",
              "loss": "ImageClassificationTask cross entropy", "preprocessing": "float32 / 127.5 - 1; NCHW; no augmentation",
              "cut_semantics": "after NCHW flatten; convolution prefix and dense suffix",
              "initial_weights_sha256": arrays_hash(weights),
              "data_sha256": arrays_hash({"train_images": train_x, "train_labels": train_y,
                                          "test_images": test_x, "test_labels": test_y}),
              "partition_sha256": arrays_hash({str(i): train_indices[p] for i, p in enumerate(partitions)}),
              "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted(output.glob("*.npz"))}}
    save_json(output / "config.json", config)
    return config


def build(backend, weights, device):
    """Load the native framework before importing any SplitFleet module."""
    w = weights
    if backend == "torch":
        import torch
        torch.set_num_threads(1)

        class CifarCNN(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 8, 3, padding=1)
                self.conv2 = torch.nn.Conv2d(8, 16, 3, padding=1)
                self.fc1 = torch.nn.Linear(1024, 32)
                self.fc2 = torch.nn.Linear(32, 10)
                with torch.no_grad():
                    for name in ("conv1", "conv2", "fc1", "fc2"):
                        layer = getattr(self, name)
                        weight = w[name + "_w"] if name.startswith("conv") else w[name + "_w"].T
                        layer.weight.copy_(torch.from_numpy(weight.copy()))
                        layer.bias.copy_(torch.from_numpy(w[name + "_b"].copy()))

            def forward(self, x):
                f = torch.nn.functional
                x = f.avg_pool2d(f.relu(self.conv1(x)), 2)
                x = f.avg_pool2d(f.relu(self.conv2(x)), 2)
                return self.fc2(f.relu(self.fc1(x.reshape(x.shape[0], 1024))))

        return CifarCNN().to(device)
    if backend == "tf":
        import tensorflow as tf
        with tf.device(device):
            # Functional models preserve variable paths when Keras deep-copies
            # each suffix replica. Sequential removes its root scope on copy,
            # changing SplitFleet's strict parameter schema identity.
            inputs = tf.keras.Input((3, 32, 32), name="images")
            features = tf.keras.layers.Permute((2, 3, 1), name="to_nhwc")(inputs)
            features = tf.keras.layers.Conv2D(8, 3, padding="same", activation="relu", name="conv1")(features)
            features = tf.keras.layers.AveragePooling2D(2, name="pool1")(features)
            features = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu", name="conv2")(features)
            features = tf.keras.layers.AveragePooling2D(2, name="pool2")(features)
            features = tf.keras.layers.Permute((3, 1, 2), name="to_nchw")(features)
            features = tf.keras.layers.Flatten(name="features")(features)
            hidden = tf.keras.layers.Dense(32, activation="relu", name="fc1")(features)
            outputs = tf.keras.layers.Dense(10, name="fc2")(hidden)
            model = tf.keras.Model(inputs, outputs, name="cifar_cnn")
            for name in ("conv1", "conv2", "fc1", "fc2"):
                weight = w[name + "_w"].transpose(2, 3, 1, 0) if name.startswith("conv") else w[name + "_w"]
                model.get_layer(name).set_weights([weight.copy(), w[name + "_b"].copy()])
        return model
    if backend == "jax":
        import jax
        import jax.numpy as jnp

        def model(params, x):
            for name in ("conv1", "conv2"):
                x = jax.lax.conv_general_dilated(x, params[name + "_w"], (1, 1), "SAME",
                                                dimension_numbers=("NCHW", "OIHW", "NCHW"))
                x = jax.nn.relu(x + params[name + "_b"][None, :, None, None])
                x = jax.lax.reduce_window(x, 0., jax.lax.add, (1, 1, 2, 2), (1, 1, 2, 2), "VALID") / 4
            x = x.reshape(x.shape[0], 1024)
            return jax.nn.relu(x @ params["fc1_w"] + params["fc1_b"]) @ params["fc2_w"] + params["fc2_b"]

        model.initial_params = {name: jnp.asarray(value) for name, value in w.items()}
        return model
    if backend == "paddle":
        import paddle
        paddle.set_device("gpu:0" if str(device).startswith("cuda") else "cpu")

        class CifarCNN(paddle.nn.Layer):
            def __init__(self):
                super().__init__()
                self.conv1 = paddle.nn.Conv2D(3, 8, 3, padding=1)
                self.conv2 = paddle.nn.Conv2D(8, 16, 3, padding=1)
                self.fc1 = paddle.nn.Linear(1024, 32)
                self.fc2 = paddle.nn.Linear(32, 10)
                for name in ("conv1", "conv2", "fc1", "fc2"):
                    layer = getattr(self, name)
                    layer.weight.set_value(w[name + "_w"])
                    layer.bias.set_value(w[name + "_b"])

            def forward(self, x):
                f = paddle.nn.functional
                x = f.avg_pool2d(f.relu(self.conv1(x)), 2)
                x = f.avg_pool2d(f.relu(self.conv2(x)), 2)
                return self.fc2(f.relu(self.fc1(x.reshape((x.shape[0], 1024)))))

        return CifarCNN()
    from tinygrad import Tensor
    Tensor.training = True

    class CifarCNN:
        def __init__(self):
            for name, value in w.items():
                parameter = Tensor(value.copy(), device=device).realize()
                parameter.requires_grad = True
                setattr(self, name, parameter)

        def __call__(self, x):
            x = x.conv2d(self.conv1_w, self.conv1_b, padding=1).relu().avg_pool2d((2, 2))
            x = x.conv2d(self.conv2_w, self.conv2_b, padding=1).relu().avg_pool2d((2, 2))
            x = x.reshape(x.shape[0], 1024)
            return (x @ self.fc1_w + self.fc1_b).relu() @ self.fc2_w + self.fc2_b

    return CifarCNN()


def tensor(backend, value, device, *, labels=False):
    if backend == "torch":
        import torch
        return torch.as_tensor(value, device=device)
    if backend == "tf":
        import tensorflow as tf
        with tf.device(device):
            return tf.convert_to_tensor(value)
    if backend == "jax":
        import jax.numpy as jnp
        return jnp.asarray(value)
    if backend == "paddle":
        import paddle
        return paddle.to_tensor(value)
    from tinygrad import Tensor
    if labels:
        value = np.asarray(value, dtype=np.int32)
    result = Tensor(value.copy(), device=device).realize()
    result.requires_grad = not labels
    return result


def batch(backend, model, images, labels, device):
    from splitfleet.tasks import ModelInputs, TaskBatch
    x = tensor(backend, images.astype(np.float32) / 127.5 - 1, device)
    y = tensor(backend, labels, device, labels=True)
    args = (model.initial_params, x) if backend == "jax" else (x,)
    return TaskBatch(ModelInputs(args), y, num_examples=len(labels))


def numpy(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value.numpy() if hasattr(value, "numpy") else value)


def optimizer(backend, model, lr):
    from splitfleet.backends import BACKEND_ADAPTERS
    opt = BACKEND_ADAPTERS.create(backend).build_optimizer(model, {"name": "sgd", "lr": lr})
    if backend == "tf":
        opt.build(model.trainable_variables)
    return opt
