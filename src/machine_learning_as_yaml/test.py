from machine_learning_as_yaml.module_generator import ModuleGenerator
import os

if __name__ == "__main__":
    mod_gen: ModuleGenerator = ModuleGenerator("../../examples/myModel2.yaml")
    model = mod_gen.construct_module()
    print(model)