import yaml
import torch.nn as nn
#from machine_learning_as_yaml.constants import ClsConstants
from typing import Any
import importlib
import sys
from copy import deepcopy

class ModuleGenerator():
    def __init__(self, config_file: str):
        self.config: dict = {}
        with open(config_file, "r") as f:
            self.config = yaml.safe_load(f)

        self.imports: dict[str, object] = {}
        self._handle_imports()

    def construct_module(self) -> nn.Module:
        module_name: str = self.config.get("model", {}).get("name", )
        backbone_module: nn.Module = self._construct_backbone()

        DynamicModuleClass: nn.Module = type(
            module_name,
            (nn.Module,),
            {
                "__init__": self.construct_init(backbone_module),
                "forward": self.construct_forward(),
            }
        )

        return DynamicModuleClass()

    def construct_forward(self) -> callable:
        def _forward(self_inst, x):
            return self_inst.backbone(x)
        return _forward

    def construct_init(self, backbone: nn.Module) -> callable:
        def _init(self_inst):
            super(type(self_inst), self_inst).__init__()
            self_inst.backbone = backbone
        return _init

    # backbone is expected to be defined as a dictionary of 1 element
    def _construct_backbone(self) -> nn.Module:
        backbone_data: dict = self.config.get("model", {}).get("structure", {}).get("backbone", {})

        for key, value in backbone_data.items():
            constructed_backbone: nn.Module = self._construct_dynamic_class(key, value)
        
        return constructed_backbone

    # All types that start with upper case are considered classes and should start with upper case letter
    def _is_class_type(self, data_name: str) -> bool:
        if len(data_name) == 0:
            return False
        return data_name[0].isupper()

    def _split_import_path(self, data_name: str) -> list:
        return data_name.strip().split(".")

    def _get_parent_import(self, data_name: str) -> str:
        return data_name.rsplit(".", 1)[0]

    def _get_class_name(self, data_name: str) -> str:
        return data_name.rsplit(".", 1)[-1]

    # Checks for nested classes, since some constructors may require dictionarys, that may not be classes
    def _has_nested_classes(self, constructor_data: dict | list) -> bool:
        if isinstance(constructor_data, list):
            has_nested: bool = False
            for value in constructor_data:
                if isinstance(value, dict):
                    for child_key, child_value in value.items():
                        if self._is_class_type(child_key):
                            has_nested = True

                        has_nested = has_nested or self._has_nested_classes(child_value)

                elif isinstance(value, list):
                    has_nested = has_nested or self._has_nested_classes(value)

                elif self._is_class_type(value):
                    return True

            return has_nested
                
        if isinstance(constructor_data, dict):
            for key in constructor_data.keys():
                if self._is_class_type(key):
                    return True

        return False

    # Each import is considerent a dict of one element
    def _handle_imports(self) -> None:
        imports: list[dict] = self.config.get("imports", [])
        for import_ in imports:
            for key, value in import_.items():
                if value is None:
                    value = key
                try:
                    self.imports[key] = importlib.import_module(value)
                except ImportError as e:
                    sys.stderr.write(f"Error while importing {value} as {key}, skipping: {e.msg}")

    # Constructs dynamically an object, that does not have any nested classes
    def _construct_dynamic_class_unnested(self, data_name: str, data: list | dict | None) -> Any:
        class_name: str = self._get_class_name(data_name)

        # check if the module is defined by another config file
        if class_name == "MLAY":
            module_generator: ModuleGenerator = ModuleGenerator(data)
            return module_generator.construct_module()

        import_path: str = self._get_parent_import(data_name)
        # If no import path is provided assume it is a torch.nn import
        if not import_path or import_path == data_name:
            import_path = "nn"
            
        # Else its some custom import, that we need to get the import of
        importer: Any = self.imports[import_path]
        if not importer:
            raise ImportError(f"Import {import_path} not found in imported modules!")

        DynamicClass: Any = getattr(importer, class_name)
        if isinstance(data, dict):

            init_args: dict = self.config.get("model", {}).get("init_args", {})
            resolved_data: dict = {}

            for key, value in data.items():
                if isinstance(value, str) and value in init_args:
                    resolved_data[key] = init_args[value]
                    continue
                resolved_data[key] = value
            return DynamicClass(**resolved_data)
        elif isinstance(data, list):
            return DynamicClass(*data)
        else:
            return DynamicClass()

    # Constructs a dynamic object, automatically resolves nesting
    def _construct_dynamic_class(self, data_name: str, data: list | dict | None) -> object:
        if not self._is_class_type(data_name):
            return data
        if not self._has_nested_classes(data):
            return self._construct_dynamic_class_unnested(data_name, data)

        # This path means the object has nested classes which need to be constructed first
        if isinstance(data, list):
            parsed_data: list = []
            for value in data:
                if isinstance(value, dict):
                    for child_key, child_value in value.items(): 
                        constructed_obj: Any = self._construct_dynamic_class(child_key, child_value)
                        parsed_data.append(constructed_obj)
                else:
                    parsed_data.append(value)

        elif isinstance(data, dict):
            parsed_data: dict = {}
            for key, value in data.items():
                if isinstance(value, dict) and len(value) == 1:
                    child_key, child_value = next(iter(value.items()))
                    constructed_child_obj: Any = self._construct_dynamic_class(child_key, child_value)
                    # parsed_data[child_key] = constructed_child_obj
                    parsed_data[key] = constructed_child_obj
                    continue

                constructed_obj: Any = self._construct_dynamic_class(key, value)
                parsed_data[key] = constructed_obj

        # After constructing nested class 
        return self._construct_dynamic_class_unnested(data_name, parsed_data)
