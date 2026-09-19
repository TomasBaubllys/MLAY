import yaml
import torch.nn as nn
from machine_learning_as_yaml.constants import ClsConstants

class ModuleGenerator():
    def __init__(self, config_file: str):
        self.config: dict = {}
        with open(config_file, "r") as f:
            self.config = yaml.safe_load(f)

    def construct_module(self) -> nn.Module:
        module_name: str = self.config.get("model", {}).get("name", )
        module: nn.Module = type()

        DynamicModuleClass: nn.Module = type(
            module_name,
            (nn.Module,),
            {
                "__init__": self.construct_init(),
                "forward": self.construct_forward()
            }
        )

        return DynamicModuleClass()

    def construct_forward(self) -> callable:
        return None

    def construct_init(self) -> callable:
        return None

    def construct_backbone(self) -> nn.Module:
        backbone_data: dict = self.config.get("model", {}).get("structure", {}).get("backbone")

        for layer in backbone_data.get()