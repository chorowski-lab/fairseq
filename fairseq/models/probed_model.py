import torch.nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


def _pick_nth(tensor_or_sequence, which=0):
    if isinstance(tensor_or_sequence, (list, tuple)):
        tensor_or_sequence = tensor_or_sequence[which]
    else:
        if which > 0:
            raise ValueError("Requested output not present")
    return tensor_or_sequence


def _detach(tensor_or_iterable):
    if isinstance(tensor_or_iterable, (list, tuple)):
        return [_detach(elem) for elem in tensor_or_iterable]
    elif isinstance(tensor_or_iterable, dict):
        return {k: _detach(v) for k, v in tensor_or_iterable.items()}
    else:
        return tensor_or_iterable.detach()


def _compile_selector(selector, default):
    if selector is None:
        return default
    elif isinstance(selector, str):
        return eval(selector)
    else:
        return selector


class Probe(torch.nn.Module):
    def __init__(
        self,
        model,
        module_name,
        backprop_to_main=False,
        output_selector=None,
        target_selector=None,
        loss_weigth=1.0,
    ):
        super().__init__()
        self._saved_tensor = None
        self._target_selector = _compile_selector(
            target_selector, default=lambda x: {"target": x}
        )
        self._loss_weigth = loss_weigth

        output_selector = _compile_selector(
            output_selector, default=lambda x: {"output": x}
        )
        hook_fn = self._get_hook(output_selector, backprop_to_main)
        self._attach(model, module_name, hook_fn)
        if backprop_to_main:
            logger.info("Registered an auxiliary loss at %s: %s", module_name, self)
        else:
            logger.info("Registered a probe at %s: %s", module_name, self)

    def _get_hook(self, output_selector, backprop_to_main):
        def hook_fn(mod, unused_inputs, outputs):
            outputs = output_selector(outputs)
            if backprop_to_main:
                self._saved_tensor = outputs
            else:
                self._saved_tensor = _detach(outputs)

        return hook_fn

    def _attach(self, model, module_name, hook_fn):
        module = dict(model.named_modules())[module_name]
        module.register_forward_hook(hook_fn)

    def compute_loss(self, minibatch):
        self._saved_tensor.update(self._target_selector(minibatch))
        ret = self(**self._saved_tensor)
        self._saved_tensor = None
        return ret


class FeedForwardProbe(Probe):
    def __init__(
        self,
        layer_dims,
        activation="torch.nn.ReLU",
        loss="torch.nn.CrossEntropyLoss",
        **kwargs,
    ):
        super().__init__(**kwargs)
        activation = eval(activation)
        in_dim, last_dim, *rest = layer_dims
        modules = [torch.nn.Linear(in_dim, last_dim)]
        for dim in rest:
            modules.append(activation())
            modules.append(torch.nn.Linear(last_dim, dim))
            last_dim = dim
        self.layers = torch.nn.Sequential(*modules)
        self.loss = eval(loss)()

    def forward(self, output, target):
        output = self.layers(output)
        return self.loss(output, target)


class Conv1DProbe(Probe):
    def __init__(self, layer_dims, kernel_size=1, activation="torch.nn.ReLU", **kwargs):
        super().__init__(**kwargs)
        activation = eval(activation)
        in_dim, last_dim, *rest = layer_dims
        assert kernel_size % 2 == 1
        modules = [
            torch.nn.Conv1d(in_dim, last_dim, kernel_size, padding=kernel_size // 2)
        ]
        for dim in rest:
            modules.append(activation())
            modules.append(
                torch.nn.Conv1d(last_dim, dim, kernel_size, padding=kernel_size // 2)
            )
            last_dim = dim
        self.layers = torch.nn.Sequential(*modules)
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, output, target, padding_mask):
        N, Cin, L = output.shape
        Nm, Cpad, Lm = padding_mask.shape
        assert Cpad == 1
        assert N == Nm
        output = F.interpolate(output, scale_factor=Lm // L)
        output = self.layers(output)
        padding_mask = padding_mask.float().squeeze(1)
        neg_mask = 1.0 - padding_mask
        target = (target * neg_mask + padding_mask * self.loss.ignore_index).long()
        loss = self.loss(output, target)
        weigth = neg_mask.sum()
        acc = (neg_mask * (torch.argmax(output, 1) == target).float()).sum() / weigth
        probe_logs = {
            "loss": loss.item(),
            "loss_weigth": weigth.item(),
            "acc": acc.item(),
            "acc_weigth": weigth.item(),
        }
        # logging.info("Probe logs: %s", probe_logs)
        return loss * self._loss_weigth, probe_logs


class ProbedModel:
    """A model which can attach small probes to analyze model behavior."""

    def _build_probe(self, cls, **kwargs):
        cls = eval(cls)
        return cls(model=self, **kwargs)

    def attach_probes(self, probe_defs):
        if not probe_defs:
            return
        self._probes = torch.nn.ModuleDict(
            {
                probe_name: self._build_probe(**probe_def)
                for probe_name, probe_def in probe_defs.items()
            }
        )

    def get_probe_losses(self, minibatch):
        loss = 0.0
        extra_log_keys = {}
        for probe_name, probe in self._probes.items():
            probe_loss, probe_log_keys = probe.compute_loss(minibatch)
            loss += probe_loss * probe._loss_weigth
            for k, v in probe_log_keys.items():
                extra_log_keys[f"probe_{probe_name}_{k}"] = v
        return loss, extra_log_keys

def reduce_probe_metrics(logging_outputs, metrics):
    handled_keys = set()
    def get_v(k):
        handled_keys.add(k)
        return sum(log.get(k, 0) for log in logging_outputs)
    for k in logging_outputs[0]:
        if k.startswith("probe_"):
            if k.endswith("_weigth"):
                continue
            v = get_v(k)
            weigth = get_v(f'{k}_weigth')
            metrics.log_scalar(k, v, weigth, round=3)
    return handled_keys