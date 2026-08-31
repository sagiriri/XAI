# IRIS-XAI — Final Results Summary

Auto-generated from the master benchmark CSV. No experiments were rerun.

## Model Performance

| dataset    | model        |   accuracy |   precision_macro |   recall_macro |   f1_macro |   auc_macro_ovr |   avg_batch_inference_time_sec |   model_size_mb |
|:-----------|:-------------|-----------:|------------------:|---------------:|-----------:|----------------:|-------------------------------:|----------------:|
| CIFAR10    | efficientnet |     0.9653 |          0.965404 |         0.9653 |   0.965294 |        0.998979 |                     0.00761456 |        15.3364  |
| CIFAR10    | resnet18     |     0.9518 |          0.952308 |         0.9518 |   0.951737 |        0.998259 |                     0.00292067 |        42.6546  |
| CIFAR10    | simplecnn    |     0.6392 |          0.700592 |         0.6392 |   0.635263 |        0.952654 |                     0.00102541 |         1.49516 |
| FUNNYBIRDS | efficientnet |     0.972  |          0.973859 |         0.972  |   0.971904 |        0.999878 |                     0.0105873  |        15.5319  |
| FUNNYBIRDS | resnet18     |     0.966  |          0.969136 |         0.966  |   0.966015 |        0.999788 |                     0.00248957 |        42.7329  |
| FUNNYBIRDS | simplecnn    |     0.964  |          0.968952 |         0.964  |   0.963855 |        0.999914 |                     0.235027   |         1.53437 |


## Explainability Score Matrix (0-100)

| dataset    | model        |   gradcam |   lime |   shap |
|:-----------|:-------------|----------:|-------:|-------:|
| CIFAR10    | efficientnet |      55.3 |   44   |   39.8 |
| CIFAR10    | resnet18     |      66.7 |   44.6 |   48.9 |
| CIFAR10    | simplecnn    |      62.4 |   40.1 |   49.7 |
| FUNNYBIRDS | efficientnet |      61.1 |   45.9 |   47.7 |
| FUNNYBIRDS | resnet18     |      70.1 |   34.8 |   63.5 |
| FUNNYBIRDS | simplecnn    |      82.1 |   48.7 |   76.4 |


## FunnyBirds Ground-Truth Comparison

| model        | xai_method   |   part_overlap_ratio |   clutter_leakage_ratio |   explanation_runtime_sec |
|:-------------|:-------------|---------------------:|------------------------:|--------------------------:|
| efficientnet | gradcam      |            0.0870871 |               0.0488937 |                 0.0544263 |
| efficientnet | lime         |            0.0565169 |               0.0773658 |                 0.907069  |
| efficientnet | shap         |            0.0596933 |               0.203989  |                14.1223    |
| resnet18     | gradcam      |            0.078762  |               0.061251  |                 0.027239  |
| resnet18     | lime         |            0.0144656 |               0.0759899 |                 0.913121  |
| resnet18     | shap         |            0.0862631 |               0.259791  |                 9.4706    |
| simplecnn    | gradcam      |            0.0976565 |               0.0919572 |                 0.155629  |
| simplecnn    | lime         |            0.0312136 |               0.0692887 |                 0.886403  |
| simplecnn    | shap         |            0.107717  |               0.178773  |                 6.93345   |


## Overall Method Ranking (Grad-CAM vs SHAP vs LIME)

| xai_method   |   avg_explainability_score |   avg_runtime_sec |   avg_part_overlap_ratio |   avg_clutter_leakage_ratio |   rank_by_score |
|:-------------|---------------------------:|------------------:|-------------------------:|----------------------------:|----------------:|
| gradcam      |                    66.287  |         0.0523959 |                0.0878352 |                   0.0673673 |               1 |
| shap         |                    54.3344 |         5.75024   |                0.0845579 |                   0.214185  |               2 |
| lime         |                    43.0167 |         0.782429  |                0.0340653 |                   0.0742148 |               3 |

