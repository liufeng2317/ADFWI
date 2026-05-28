"""Inversion-history cache helpers shared by FWI drivers.

The helpers in this module only move tensors into the historical Python lists
used by ``AcousticFWI`` and ``ElasticFWI``. They do not decide which physical
parameters belong to a given inversion class, when a result should be plotted,
or how cached arrays should be interpreted.
"""


def tensor_to_numpy(tensor):
    """Return an immutable detached CPU NumPy snapshot for a model tensor."""
    return tensor.cpu().detach().numpy().copy()


def append_epoch_loss(cache_owner, loss_epoch):
    """Append one epoch loss to an FWI driver's ``iter_loss`` history."""
    cache_owner.iter_loss.append(loss_epoch)


def should_cache_epoch(epoch_id, cache_result_epoch):
    """Match the historical modulo rule for cached model snapshots."""
    return epoch_id % cache_result_epoch == 0


def snapshot_model_parameters(model, parameter_names):
    """Collect detached NumPy snapshots for existing model parameters."""
    snapshots = {}
    for name in parameter_names:
        parameter = getattr(model, name, None)
        if parameter is not None:
            snapshots[name] = tensor_to_numpy(parameter)
    return snapshots


def append_model_snapshots(cache_owner, snapshots, epoch_id):
    """Append model snapshots to ``iter_<parameter>`` lists and record epoch."""
    for name, snapshot in snapshots.items():
        getattr(cache_owner, f"iter_{name}").append(snapshot)
    cache_owner.cache_iter_index.append(epoch_id)


def append_required_gradient_snapshots(cache_owner, model, parameter_names):
    """Append gradients for parameters that the model marks as trainable."""
    snapshots = {}
    for name in parameter_names:
        if model.get_requires_grad(name):
            gradient = tensor_to_numpy(getattr(model, name).grad)
            getattr(cache_owner, f"iter_{name}_grad").append(gradient)
            snapshots[name] = gradient
    return snapshots
