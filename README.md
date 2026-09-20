# Machine Learning as Yaml
This project aims to implement dynamic machine learning model creation for PyTorch through configuration files

## Why?
While doing research experiments with PyTorch I have realized that creating a different model/module for each change takes a lot of boiler plate code, which can get messy quite fast. However *yaml* files are a lot more readable and manageable that *python* code.

## Setup guide
To make the setup easier, the entire generator lives in a single file named **module_generator.py**.
So you can either:
- clone this repository and import it to your project.
- copy the contents of **module_generator.py** and paste it to you desired file

## How to use?
To use the generator, you must first import it
```
from module_generator import ModuleGenerator
```

Then create an instance and provide a configuration file
```
module_generator = ModuleGenerator("your_file_path.yaml")
```

Once thats done, you can retrieve the created dynamic class with
```
MyCustomModelClass = module_generator.get_module_class()
```

And then create an instance of your model just like you normally would with PyTorch
```
my_custom_model = MyCustomModelClass(<args>)
```

## Documentation
**custom imports**
To import some custom library, you have to provide them in the file under *imports*. The syntax is as follows:
```
imports:
    # equivalent to "import numpy as np"
    - np: numpy

    # equivalent to "import matplotlib.pyplot as plt"
    - plt: matplotlib.pyplot

```
Keep in mind that by default torch and torch.nn as nn are imported by default

**defining constants**
Constants can be defined under *constants* attribute, however their usage is not fully tested:
```
constants:
    a: 2
    b: "hello world!"
```

**defining a model**
To define a model you need to use the "model" attribute, the basic structure is as follows:
```
model:
    # "name" attribute will become the internal name of your class
    name: "MySuperCoolModel"
    
    # "init_args" is used to provide which initialization arguments should be provided when constructing your model, they will also be set as model attributes during __init__. For now init_args only supports basic python types
    init_args:
        num_classes: 10
        skip_train: True
        key: value

    # custom initialization code can be provided, if your model needs extra complexity, it will be the last thing executed if "replacements" attribute is not provided. the "init_code" must be valid python code
    init_code: |
        if num_classes > 10:
            self.number = 123
        else:
            self.number = 456

    
    # "structure" attribute is used to provide declaration of your models structure. As you can see you will still need to read PyTorch documentation to know which arguments to provide. All modules constructed under "structere" will be available through self.<module_name>

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
                # You can also other configuration files inside to defined custom modules. That is done by used a special "MLAY__" keyword
                - MLAY__: "path_to_another_mlay_config.yaml"
        classifier:
            Linear:
                in_features: num_classes
                out_features: num_classes
        custom_block:
            Linear:
                in_features: 64
                out_features: 64

```


## Examples
