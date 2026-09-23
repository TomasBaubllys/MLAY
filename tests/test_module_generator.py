from machine_learning_as_yaml.module_generator import ModuleGenerator
import torch
import torch.nn as nn
import os

def test_basic_model():
    config: str = """
    model:
        name: basic_model
        structure:
            backbone:
                Sequential:
                    - Conv2d:
                        in_channels: 3
                        out_channels: 16
                        kernel_size: 3
                        stride: 2
                        padding: 1
                    - BatchNorm2d:
                        num_features: 16
                    - ReLU:
                        inplace: True
                    - Conv2d:
                        in_channels: 16
                        out_channels: 32
                        kernel_size: 3
                        stride: 2
                        padding: 1
                    - BatchNorm2d:
                        num_features: 32
                    - ReLU:
                        inplace: True
                    - AdaptiveAvgPool2d:
                        output_size: [ 1, 1 ]
                    - Flatten:
                    - Linear:
                        in_features: 32
                        out_features: 10
            classifier:
                Linear:
                    in_features: 10
                    out_features: 10
        forward:
            - backbone
            - classifier

        """
    file_name: str = "test_basic_model.yaml" 
    with open(file_name, "w+") as f:
        f.write(config)
    module_generator: ModuleGenerator = ModuleGenerator(file_name)
    BasicModelClass: type = module_generator.get_module_class()
    model: nn.Module = BasicModelClass()

    test_tensor: torch.Tensor = torch.rand(1, 3, 32, 32)
    output = model(test_tensor)
    assert output.shape == (1, 10)

    os.remove(file_name)

if __name__ == "__main__":
    test_basic_model()