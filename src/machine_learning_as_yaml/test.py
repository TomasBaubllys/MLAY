from machine_learning_as_yaml.module_generator import ModuleGenerator
import os
import inspect

if __name__ == "__main__":
    mod_gen: ModuleGenerator = ModuleGenerator("../../examples/myModel2.yaml")
    Model = mod_gen.get_module_class()
    model = Model(num_classes = 10)
    print(model)
    print(model.number)