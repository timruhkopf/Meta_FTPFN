from typing import Mapping

import torch
import torch.nn as nn


class MyModuleList(nn.Module):
    def __init__(self, modules):
        super().__init__()
        self.modules_list = nn.ModuleList(modules)

    def forward(self, x, *args, **kwargs):
        for module in self.modules_list:
            x = module(x, *args, **kwargs)
        return x


class HierarchicalPFN(nn.Module):
    def __init__(
        self, frozen_model: nn.Module, interleaved_layers: Mapping[str, nn.Module]
    ):
        """Initialize the InterleavedModel with a frozen model and interleaved layers.

        Args:
            frozen_model (nn.Module): The pre-trained frozen model.
            interleaved_layers (Dict[str, nn.Module]): A dictionary mapping target module names
                in the frozen model to their corresponding interleaved layers.

        """

        super().__init__()
        self.frozen_model = frozen_model

        for param in self.frozen_model.parameters():
            param.requires_grad = False  # Freeze the pre-trained model

        self.interleaved_layers = interleaved_layers

        # verify that all target modules exist in the frozen model
        for name in interleaved_layers.keys():
            if name not in dict(self.frozen_model.named_modules()):
                raise ValueError(
                    f"Target module '{name}' not found in the frozen model."
                )

        # wrap frozen model layers with the interleaved_layers
        for name, module in self.frozen_model.named_modules():
            if name in self.interleaved_layers:
                interleaved_layer = self.interleaved_layers[name]
                # Replace the target module with a sequential module
                wrapped_module = MyModuleList(
                    [  # this way, we can intercept any layer including all arguments of the call
                        interleaved_layer,
                        module,
                    ]
                )
                # Set the wrapped module back to the frozen model under the same name
                parent_module = self._get_parent_module(self.frozen_model, name)
                setattr(parent_module, name.split(".")[-1], wrapped_module)

        self._single_eval_pos = None

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

    def _get_parent_module(self, model: nn.Module, module_name: str) -> nn.Module:
        """Get the parent module of a given module by its name."""
        parts = module_name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        return parent

    def forward(self, x, **kwargs):
        self.single_eval_pos = kwargs[
            "single_eval_pos"
        ]  # communicate to interleaved layers (should the pfn not pass the argument to the layer explicitly! (e.g. intercepting a linear))
        value = self.frozen_model(x, **kwargs)
        self.single_eval_pos = None  # reset after forward
        return value

    def predict(self, target_task, related_tasks, **kwargs):
        """Predict using the frozen model with interleaved layers."""
        # Combine target and related tasks in batch dimension
        raise NotImplementedError(
            "predict method needs to be implemented based on specific use case."
        )
        combined_input = torch.cat([target_task.repeat(), related_tasks], dim=1)
        return self.frozen_model(combined_input, **kwargs)


if __name__ == "__main__":
    from ppfn.model.ppfn.ft_ppfn import load_frozen_model
    from ppfn.model.mymodel.cross_fusion import CrossFusion

    frozen_model = load_frozen_model()
    interleaved_layers = {
        "transformer_encoder.layers.0.linear1": CrossFusion(d_model=512, num_heads=8),
        "transformer_encoder.layers.2.linear1": CrossFusion(d_model=512, num_heads=8),
    }
    model = HierarchicalPFN(
        frozen_model=frozen_model, interleaved_layers=interleaved_layers
    )
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
    x_target_eval = x[:, :1, :] # query
    y_target_eval = y[:, :1]

    x_related_eval = x[:, 1:, :] # key, value
    # we want to have the exact same query location!
    x_related_eval[:single_eval_pos, :, :] = x_target_eval[:single_eval_pos, :1, :]
    y_related_eval = y[:, 1:]

    # concat in batch dimension
    # target task, unconditional related tasks, conditional related tasks
    # q, k, v
    x_eval = torch.cat([x_target_eval, x_related_eval, x_related_eval], dim=1)
    y_eval = torch.cat([y_target_eval, y_related_eval, y_related_eval], dim=1)

    model.eval()
    with torch.no_grad():
        _ =  model((x_eval, y_eval), single_eval_pos=single_eval_pos)
   
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
    x_related  = x
    y_related  = y

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

    _ =  model((x_train, y_train), single_eval_pos=single_eval_pos)



    print("success")
