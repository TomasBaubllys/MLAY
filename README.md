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
```python
from module_generator import ModuleGenerator
```

Then create an instance and provide a configuration file
```python
module_generator = ModuleGenerator("your_file_path.yaml")
```

Once thats done, you can retrieve the created dynamic class with
```python
MyCustomModelClass = module_generator.get_module_class()
```

And then create an instance of your model just like you normally would with PyTorch
```python
my_custom_model = MyCustomModelClass(<args>)
```

## Documentation

A config file is made up of a few top-level attributes: `imports`, `constants`, and `model`. Below is a breakdown of each one and everything they contain.

### `imports`
Used to import any custom library your model needs. By default `torch` and `torch.nn` (as `nn`) are already imported for you, so you don't need to declare those unless you want them under a different name.

```yaml
imports:
    # equivalent to "import numpy as np"
    - np: numpy

    # equivalent to "import matplotlib.pyplot as plt"
    - plt: matplotlib.pyplot
```

The key is the alias you want to use inside your config, and the value is the actual module path being imported.

### `constants`
Lets you define reusable values that can be referenced elsewhere in the file. Usage is not fully tested yet, so treat this as experimental.

```yaml
constants:
    a: 2
    b: "hello world!"
```

### `model`
This is the main attribute where your model is actually defined. It has several sub-attributes:

#### `name`
The name that will be given to your generated class internally.

```yaml
model:
    name: "MySuperCoolModel"
```

#### `init_args`
Defines which arguments your model's `__init__` should accept. Each one is also automatically set as an attribute on the model instance (`self.<arg_name>`) during initialization. For now, `init_args` only supports basic Python types.

```yaml
init_args:
    num_classes: 10
    skip_train: True
    key: value
```

#### `init_code`
Optional custom initialization code, for when your model needs extra logic beyond just declaring its structure. Must be valid Python code. If `replace` is not provided, this is the last thing executed inside `__init__`.

```yaml
init_code: |
    if num_classes > 10:
        self.number = 123
    else:
        self.number = 456
```

#### `structure`
Declares your model's layers/modules. You'll still need to check the PyTorch docs to know which arguments each module expects. Every module declared here becomes accessible through `self.<module_name>`.

```yaml
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

            # You can nest other config files inside using the special "MLAY__" keyword
            - MLAY__: "path_to_another_mlay_config.yaml"

            # If you imported a custom library (e.g. cs: my_custom_lib), you can reference its classes here too
            - cs.MyCustomClass:
                some_argument: some_value

    classifier:
        Linear:
            in_features: num_classes
            out_features: num_classes
```

A few things worth calling out:
- Values under a layer (like `in_channels`, `num_features`, etc.) map directly to that PyTorch module's constructor arguments.
- `MLAY__` lets you compose models out of other `.mlay` YAML configs, so you can build re-usable sub-models and drop them into a bigger one.
- Named entries (`backbone`, `classifier`, `custom_block`, etc.) become named submodules on your model, so they show up as `self.backbone`, `self.classifier`, and so on.

#### `forward` / `forward_code`
Defines how data flows through your model.

- `forward` is the simple option — a list of module names to call in sequence, passing the output of each into the next.

```yaml
forward:
    - backbone
    - classifier
```

- `forward_code` is the advanced option — write the `forward` method body yourself in raw Python, which is useful when you need branching, skip connections, multiple inputs, etc.

```yaml
forward_code: |
    y = self.backbone(x)
    y = self.custom_block(x)
    return x + y
```

Use whichever fits your model — `forward` for straightforward sequential flow, `forward_code` when you need more control.

#### `replace`
Lets you swap out a specific layer inside your structure after everything else has been built. This runs last, after `structure` and `init_code`, so it's useful for patching layers coming from a nested `MLAY__` config without having to redefine the whole thing.

```yaml
replace:
    - backbone:
        # "index" can be a list of indexes (useful when backbone is a Sequential, etc.)
        # or a name, in which case it replaces the first match
        index: [1, 1]
        with: Conv2d
        kwargs:
            in_channels: 3
            out_channels: 16
            kernel_size: 5
            stride: 2
            padding: 3
```

- `with` is the class to replace the target layer with.
- `kwargs` are the constructor arguments passed to that new layer.
- `index` can mix numeric indexes and names to walk through nested structures (e.g. `[0, "backbone", "Conv2d"]` to reach into a nested `MLAY__` submodule).

## Examples
More complete, ready-to-run examples — including nested models composed with `MLAY__` and layer patching with `replace` — are available in the [`examples/`](./examples) folder.