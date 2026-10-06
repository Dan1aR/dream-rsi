# KernelBench problem files

The `init_program.py` in each directory is the unmodified KernelBench problem
listed below, taken from
[ScalingIntelligence/KernelBench](https://github.com/ScalingIntelligence/KernelBench)
at commit `423217d9fda91e0c2d67e4a43bf62f96f6d104f1`. KernelBench is
distributed under the MIT License (see [LICENSE](LICENSE)); it is **not**
covered by the Apache 2.0 license of the rest of this repository.

| Directory | KernelBench problem |
| --- | --- |
| `kb_l3_11_vgg16` | `KernelBench/level3/11_VGG16.py` |
| `kb_l1_40_layernorm` | `KernelBench/level1/40_LayerNorm.py` |
| `kb_l2_8_convdiv` | `KernelBench/level2/8_Conv3d_Divide_Max_GlobalAvgPool_BiasAdd_Sum.py` |
| `kb_l2_43_convmax` | `KernelBench/level2/43_Conv3d_Max_LogSumExp_ReLU.py` |

Each file seeds the search of the matching `tasks/kernelbench/kb_*` task and is
the reference model that `tasks/kernelbench/server/launch_server.py` scores
candidates against.
