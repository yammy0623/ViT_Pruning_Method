# Experiment

## CIFAR10

| Experiment Name               | Original Throughput (im/s) | Original Accuracy | ToMe Throughput (im/s) | ToMe Accuracy | Epochs | Batch Size | Learning Rate |
|-------------------------------|----------------------------|-------------------|-------------------------|---------------|--------|------------|---------------|
| efficientnet_b0.ra_in1k       | 732.41                    | 95.55%           | -                       | -             | 200    | 64         | 0.0001        |
| mixnet_s.ft_in1k              | 330.78                    | 95.83%           | -                       | -             | 200    | 32         | 0.0001        |
| mobilenetv3_large_100.ra_in1k | 362.32                    | 95.35%           | -                       | -             | 200    | 32         | 0.0001        |
| deit_tiny_distilled           | 486.30                    | 96.31%           | 639.70                  | 96.31%        | 300    | 64         | 0.0001        |
| deit3                         | 125.30                    | 97.20%           | 54.37                   | 8.01%         | 200    | 32         | 0.0001        |
| augreg                        | 902.22                    | 94.99%           | 577.79                  | 94.99%        | 200    | 64         | 0.0001        |
| cait                          | 147.05                    | 95.99%           | 297.76                  | 95.99%        | 200    | 32         | 0.0001        |

## ImageNet

| Experiment Name               | Original Throughput (im/s) | Original Accuracy | ToMe Throughput (im/s) | ToMe Accuracy | Epochs | Batch Size | Learning Rate |
|-------------------------------|----------------------------|-------------------|-------------------------|---------------|--------|------------|---------------|
| efficientnet_b0.ra_in1k       |                     |            | -                       | -             | 200    | 64         | 0.0001        |
| mixnet_s.ft_in1k              |                     |            | -                       | -             | 200    | 32         | 0.0001        |
| mobilenetv3_large_100.ra_in1k |                     |            | -                       | -             | 200    | 32         | 0.0001        |
| deit_tiny_distilled           |                     |            |                   |         | 300    | 64         | 0.0001        |
| deit3                         |                     |            |                    |          | 200    | 32         | 0.0001        |
| augreg                        |                     |            |                   |         | 200    | 64         | 0.0001        |
| cait                          |                     |            |                   |         | 200    | 32         | 0.0001        |
