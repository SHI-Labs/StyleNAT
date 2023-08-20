# Hydra-core configurations

We use [hydra](https://hydra.cc/docs/intro/) to track our hyper-parameters.
`runs` directory contains settings for the runs that were used for each dataset

`meta_conf.yaml` is an explanation of all the possible arguments, the type, and
what they are used in.

`inference.yaml` is a simple inference example configuration.

`conf.yaml` is the bare minimum configuration for training.

Here's a quick reference on hydra's override syntax

Overwriting an existing value
```
arg_key=new_vale
nested/key=new_value
```

Add a new argument
```
+new_arg=new_value
+new/nested/arg=new_value
```

Remove an argument
```
~arg_key
~arg_key=value
~nested/arg
~nested/arg=value
```

Use a different base config file
```
python main.py --config-name attn_map
python main.py -cn inference
```
