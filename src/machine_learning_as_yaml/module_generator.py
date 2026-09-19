import yaml
import torch
import torch.nn as nn
from typing import Any, Type
import importlib
import sys

from enum import Enum

class ModuleGeneratorConstants(Enum):
    # This name is reserved for including other config files inside your config file as generators
    CONFIG_CLASS_NAME: str = "MLAY__"

class ModuleGenerator:
    def __init__(self, config_file: str):
        self.config: dict = {}
        with open(config_file, "r") as f:
            self.config = yaml.safe_load(f)

        self.imports: dict[str, object] = {}
        self._handle_imports()
        self.ConstructedClass: nn.Module = None

    def _construct_class(self) -> Type[nn.Module]:
        module_name: str = self.config.get("model", {}).get("name", )
        computational_modules: dict = {}

        for name, value in self.config.get("model", {}).get("structure", {}).items():
            if isinstance(value, dict) and len(value) == 1:
                class_key, class_value = next(iter(value.items()))
                computational_modules[name] = self._construct_dynamic_class(class_key, class_value)

        DynamicModuleClass: nn.Module = type(
            module_name,
            (nn.Module,),
            {
                "__init__": self._construct_init(computational_modules),
                "forward": self._construct_forward(),
            }
        )

        self.ConstructedClass = DynamicModuleClass
        return DynamicModuleClass

    def get_module_class(self) -> type: 
        if self.ConstructedClass is not None:
            return self.ConstructedClass

        return self._construct_class()

    def _construct_forward(self) -> callable:
        # Case when forward is provided as set of modules
        forward_dec: list[str] = self.config.get("model", {}).get("forward", None)
        if forward_dec is not None:
            def _forward(self_inst, x):
                for comp_mod in forward_dec:
                    module: nn.Module = getattr(self_inst, comp_mod)
                    x = module(x)
                return x
                
            return _forward

        # Case when forward is provided as a raw string
        forward_dec_code: str = self.config.get("model", {}).get("forward_code", None)
        if forward_dec_code is not None:
            local_objects: dict = {}
            full_code: str = "def _forward(self, x: torch.Tensor) -> torch.Tensor:\n"
            for line in forward_dec_code.strip().split("\n"):
                full_code += f"  {line}\n"
            exec(full_code, self.imports, local_objects)
            return local_objects["_forward"]

        raise ValueError("Either 'forward' (list) or 'forward_code' (string) must be defined in the YAML model config!")
        
    def _construct_init(self, modules: dict[str, nn.Module]) -> callable:
        init_args: dict = self._get_init_args()
        init_code: str = self.config.get("model", {}).get("init_code", None)

        def _init(self_inst, *args, **kwargs):
            super(type(self_inst), self_inst).__init__()
            for module_name, module in modules.items():
                setattr(self_inst, module_name, module)

            final_args: dict = {**init_args, **kwargs}

            for key, value in final_args.items():
                setattr(self_inst, key, value)

            if init_code:
                full_code: str = "def _custom_init(self):\n"
                for line in init_code.strip().split("\n"):
                    full_code += f"    {line}\n"

                local_bindings: dict = {}
                exec(full_code, self.imports, local_bindings)
                custom_init: callable = local_bindings["_custom_init"]
                custom_init(self_inst)

            # Handle replacements
            replacements: list = self._get_replace_list()
            for replacement in replacements:
                module_name, replacement_args = next(iter(replacement.items()))
                if not hasattr(self_inst, module_name):
                    sys.stderr.write(f"Module {module_name} not found in constructed class")

                module: nn.Module = getattr(self_inst, module_name)

                # Traverse throught the indexes except for the last one
                temp_module: nn.Module = module
                replacement_indexes: list = replacement_args.get("indexes", [])
                for index in replacement_indexes[:-1]:
                    if hasattr(temp_module, "__getitem__"):
                        temp_module = temp_module[index]
                    elif hasattr(temp_module, "features"):
                        temp_module = temp_module.feature[index]
                    else:
                        temp_module = getattr(temp_module, str(index))

                # construct the module
                replacement_name: str = replacement_args.get("with")
                replacement_kwargs: str = replacement_args.get("kwargs", {}) 
                replacement_object: Type[nn.Module] = self._construct_dynamic_class(replacement_name, replacement_kwargs)

                # replace it
                final_idx: int | str = replacement_indexes[-1]
                if hasattr(temp_module, "__setitem__"):
                    temp_module[final_idx] = replacement_object
                elif hasattr(temp_module, "features"):
                    temp_module.features[replacement_indexes[-1]] = replacement_object
                else:
                    setattr(temp_module, str(final_idx), replacement_object)

        return _init

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
        if class_name == ModuleGeneratorConstants.CONFIG_CLASS_NAME.value:
            module_generator: ModuleGenerator = ModuleGenerator(data)
            SubModuleClass: Type[nn.Module] = module_generator.get_module_class()
            submodule_object: nn.Module = SubModuleClass(**module_generator._get_init_args())
            return submodule_object

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

            constants: dict = self._get_constants()
            resolved_data: dict = {}

            for key, value in data.items():
                if isinstance(value, str) and value in constants:
                    resolved_data[key] = constants[value]
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

    def _get_constants(self) -> dict:
        return self.config.get("constants", {})

    def _get_init_args(self) -> dict:
        return self.config.get("model", {}).get("init_args", {})

    def _get_replace_list(self) -> dict:
        return self.config.get("model", {}).get("replace", [])