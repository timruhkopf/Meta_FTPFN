from typing import List, Mapping, Union

import torch
import torch.nn as nn

from pfns4hpo.priors.prior import Batch

from ifbo.transformer import TransformerModel
import logging

from ppfn.model.baselines.ft_pfn_padding import PaddableTransformerModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# TODO move to utils
class MySequential(nn.Sequential):
    """
    Acts like nn.Sequential but propagates all extra arguments
    (like single_eval_pos) to every sub-module.
    """

    def forward(self, x, *args, **kwargs):
        for module in self:
            # We pass x AND all extra args to every layer in the wrapper
            x = module(x, *args, **kwargs)
        return x


class HierarchicalPFN(nn.Module):
    def __init__(
        self,
        frozen_model: nn.Module,
        interleaved_layers: Union[Mapping[str, nn.Module], List],
        pass_hp_to_interleaved: bool = True,
    ):
        """Initialize the InterleavedModel with a frozen model and interleaved layers.

        Args:
            frozen_model (nn.Module): The pre-trained frozen model.
            interleaved_layers (Dict[str, nn.Module]): A dictionary mapping target module names
                in the frozen model to their corresponding interleaved layers.

        """

        super().__init__()
        self.frozen_model = frozen_model
        self.paddable = True
        self.pass_hp_to_interleaved = pass_hp_to_interleaved

        # check if frozen_model knows the src_key_padding_mask argument, if not, we need to make sure to propagate it to the interleaved layers
        if isinstance(self.frozen_model, TransformerModel):
            logger.info(
                "Frozen model is a TransformerModel, will not be able to propagate src_key_padding_mask to interleaved layers if needed."
            )
            self.paddable = False


        for param in self.frozen_model.parameters():
            param.requires_grad = False  # Freeze the pre-trained model

        # avoiding an OmegaConf instantiation / naming issue, we allow to pass a list of dicts
        if isinstance(interleaved_layers, list):
            self.interleaved_layers = dict()
            # convert to dict with layer names as keys
            for item in interleaved_layers:
                name = item["name"]
                layer = item["layer"]
                self.interleaved_layers[name] = layer
        else:
            self.interleaved_layers = interleaved_layers

        self._params_registry = nn.ModuleList(self.interleaved_layers.values())

        # verify that all target modules exist in the frozen model
        for name in self.interleaved_layers.keys():
            if name not in dict(self.frozen_model.named_modules()):
                raise ValueError(
                    f"Target module '{name}' not found in the frozen model."
                )

        # wrap frozen model layers with the interleaved_layers
        for name, module in self.frozen_model.named_modules():
            if name in self.interleaved_layers:
                interleaved_layer = self.interleaved_layers[name]
                # Replace the target module with a sequential module
                wrapped_module = MySequential(
                    # Notice, that we cannot use nn.Sequential here, because it would only pass on the first argument,
                    # ignoring e.g. single_eval_pos or others that might be needed by the interleaved layer.
                    *[  # this way, we can intercept any layer including all arguments of the call
                        interleaved_layer,
                        module,
                    ]
                )
                # Set the wrapped module back to the frozen model under the same name
                parent_module = self._get_parent_module(self.frozen_model, name)
                setattr(parent_module, name.split(".")[-1], wrapped_module)

        # to not break interface with the pfn, we push these side attributes into the wrapper model, and propagate them to the interleaved layers as needed
        self._single_eval_pos = None
        self._hp = None


    @property
    def single_eval_pos(self):
        return self._single_eval_pos

    @single_eval_pos.setter
    def single_eval_pos(self, value):
        self._single_eval_pos = value
        # propagate to interleaved layers
        for layer in self.interleaved_layers.values():
            if hasattr(layer, "single_eval_pos"):
                layer.single_eval_pos = value

    @property
    def hp(self):
        return self._hp

    @hp.setter
    def hp(self, value):
        self._hp = value
        # propagate to interleaved layers
        for layer in self.interleaved_layers.values():
            if hasattr(layer, "hp"):
                layer.hp = value

    def _get_parent_module(self, model: nn.Module, module_name: str) -> nn.Module:
        """Get the parent module of a given module by its name."""
        parts = module_name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        return parent

    def forward(self, batch: Batch, single_eval_pos=None, hp=None, **kwargs):

        # parse input to meet ft_pfn format
        x = (batch.x, batch.y)
        single_eval_pos = (
            batch.single_eval_pos if single_eval_pos is None else single_eval_pos
        )

        if single_eval_pos is None:
            raise ValueError(
                "single_eval_pos must be provided in the batch or as an argument."
            )

        # propagate to interleaved layers to not break the pfn interface

        if self.pass_hp_from_frozen:
            # hp by convention will give us the information of all locations in the sequence!
            hp = self.frozen_model.encoder.configuration_enc(batch.x[:, :, 2:])
            self.hp = hp

        kwargs.pop("hp", None)  # remove hp from kwargs to avoid passing it to the frozen model
        self.single_eval_pos = single_eval_pos

        kwargs["single_eval_pos"] = single_eval_pos
        if not self.paddable and "src_key_padding_mask" in kwargs:

            kwargs.pop("src_key_padding_mask")

        value = self.frozen_model(x, **kwargs)

        # reset the attributes to avoid side-effects
        self.single_eval_pos = None  # reset after forward
        self.hp = None

        # TODO the value here can be intercepted by model callbacks to log metrics on
        #  the three streams, when we move towards an aggregation strategy!

        return value

    def predict(self, target_task, related_tasks, **kwargs):
        """Predict using the frozen model with interleaved layers."""
        # Combine target and related tasks in batch dimension
        raise NotImplementedError(
            "predict method needs to be implemented based on specific use case."
        )
        combined_input = torch.cat([target_task.repeat(), related_tasks], dim=1)
        return self.frozen_model(combined_input, **kwargs)

    def state_dict(self, *args, **kwargs):
        """
        Gathers state_dicts from all interleaved layers into a nested dictionary.
        """
        return {
            name: layer.state_dict(*args, **kwargs)
            for name, layer in self.interleaved_layers.items()
        }

    def load_state_dict(self, state_dict: dict, strict: bool = True):
        """
        Distributes the nested state_dict back to the individual layers.
        """
        # Clean the state_dict keys if they come from a DDP save
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        for name, layer_state in state_dict.items():
            if name in self.interleaved_layers:
                self.interleaved_layers[name].load_state_dict(
                    layer_state, strict=strict
                )
            elif strict:
                # If we are loading a full trainer checkpoint, it might have
                # other keys; we only care about the keys in our registry.
                logger.warning(
                    f"Key '{name}' in state_dict not found in interleaved_layers."
                )

    def save(self, path: str):
        """Saves only the relevant interleaved weights."""
        torch.save(self.state_dict(), path)

    def load(self, path: str, strict: bool = True):
        """Loads weights from a file into the current instance."""
        state_dict = torch.load(path, map_location="cpu")
        self.load_state_dict(state_dict, strict=strict)

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        frozen_model: nn.Module,
        interleaved_layers: Union[Mapping[str, nn.Module], List],
    ):
        """Factory method: Creates the wrapper and loads weights immediately."""
        instance = cls(frozen_model, interleaved_layers)
        instance.load(path)  # Calls the instance method above
        return instance


if __name__ == "__main__":
    from ppfn.utils.load_ftpfn import load_frozen_model
    from ppfn.model.mymodel.cross_fusion import CrossFusion

    frozen_model = load_frozen_model()
    interleaved_layers = {
        "transformer_encoder.layers.0.linear1": CrossFusion(d_model=512, num_heads=8),
        "transformer_encoder.layers.2.linear1": CrossFusion(d_model=512, num_heads=8),
    }
    model = HierarchicalPFN(
        frozen_model=frozen_model, interleaved_layers=interleaved_layers
    )

    model.state_dict()
    print(model)

    def dummy_ft_batch(T=32, B=8, D=5, Tsplit=25):
        """
        Create a dummy batch of input data.

        returns:
            x: tuple of (features, targets)
            Tsplit: int, split index between train and test
        """
        torch.manual_seed(42)

        assert D >= 1 and D <= 11  # 1 int + up to 10 float features

        # First feature: integer in [0, 1000]
        ints = torch.randint(low=0, high=1001, size=(T, B))  # [T, B, 1][web:1]
        floats = torch.rand(T, B, D)  # [T, B, D-1][web:5]

        x_train = floats[:Tsplit]
        x_test = floats[Tsplit:]
        y_train = ints[:Tsplit].float()

        # this is the target format:
        tokens = (torch.cat([x_train, x_test], dim=0), y_train)
        # single_eval_pos=x_train.shape[0],
        # src_key_padding_mask=None

        return tokens, Tsplit

    (x, y), single_eval_pos = dummy_ft_batch()

    # test batch: ----------------------------

    n_related = x.shape[1] - 1
    x_target_eval = x[:, :1, :]  # query
    y_target_eval = y[:, :1]

    x_related_eval = x[:, 1:, :]  # key, value
    # we want to have the exact same query location!
    x_related_eval[:single_eval_pos, :, :] = x_target_eval[:single_eval_pos, :1, :]
    y_related_eval = y[:, 1:]

    # concat in batch dimension
    # unconditional target task, unconditional related tasks, target tasks to be conditioned on the related tasks
    # q, k, v
    x_eval = torch.cat(
        [x_target_eval, x_related_eval, x_target_eval.expand(-1, n_related + 1, -1)],
        dim=1,
    )
    y_eval = torch.cat(
        [y_target_eval, y_related_eval, y_target_eval.expand(-1, n_related + 1)], dim=1
    )

    model.eval()
    with torch.no_grad():
        _ = model((x_eval, y_eval), single_eval_pos=single_eval_pos)

    # n_related = x.shape[1] - 1
    # x_target_eval = x[:, :1, :].repeat(1, n_related +1, 1) # query
    # y_target_eval = y[:, :1].repeat(1, n_related +1)

    # x_related_eval = x[:, 1:, :] # key, value
    # # we want to have the exact same query location!
    # x_related_eval[:single_eval_pos, :, :] = x_target_eval[:single_eval_pos, :1, :]
    # y_related_eval = y[:, 1:]

    # # concat in batch dimension
    # # target task, unconditional related tasks, conditional related tasks
    # # q, k, v
    # x_eval = torch.cat([x_target_eval, x_related_eval], dim=1)
    # y_eval = torch.cat([y_target_eval, y_related_eval], dim=1)

    # model.eval()
    # with torch.no_grad():
    #     _ =  model((x_eval, y_eval), single_eval_pos=single_eval_pos)

    # train_batch ----------------------------
    x_related = x
    y_related = y

    x_target = x.roll(1, dims=1)  # shift by one along sequence dimension
    y_target = y.roll(1, dims=1)
    # TODO what about adding in unrelated examples to foster ignoring when not useful.
    # TODO what about taking the same task and just distorting it with e.g. an affine prior
    # TODO pairs must have the same test locations!

    # concat in batch dimension
    # target task, unconditional related tasks, conditional related tasks
    x_train = torch.cat([x_target, x_related, x_related], dim=1)
    y_train = torch.cat([y_target, y_related, y_related], dim=1)

    model.train()

    _ = model((x_train, y_train), single_eval_pos=single_eval_pos)

    print("success")
