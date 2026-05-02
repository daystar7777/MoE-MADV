# DeepSeek Q4 Performance Matrix

Generated: 2026-05-02T16:39:00

## Inference

| case | prompt | repeat | prewarm | madvise | gap | chunk | wall s | I/O active | disk GiB | prompt t/s | gen t/s | prefill I/O | decode I/O |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no_prewarm_madvise_off | decode_json_seed | 1 | 0 | 0 | 4.00 | 1 | 120.34 | 99.3% | 173.66 | 0.80 | 1.10 | 100.0% | 91.6% |
| no_prewarm_madvise_off | decode_json_seed | 2 | 0 | 0 | 4.00 | 1 | 111.72 | 99.3% | 147.93 | 0.80 | 1.10 | 100.0% | 92.9% |
| no_prewarm_madvise_off | decode_json_seed | 3 | 0 | 0 | 4.00 | 1 | 114.94 | 99.6% | 146.55 | 0.80 | 1.00 | 100.0% | 97.1% |
| no_prewarm_madvise_off | decode_plain_seed | 1 | 0 | 0 | 4.00 | 1 | 113.87 | 99.7% | 143.76 | 0.70 | 0.90 | 100.0% | 98.6% |
| no_prewarm_madvise_off | decode_plain_seed | 2 | 0 | 0 | 4.00 | 1 | 116.89 | 99.7% | 151.02 | 0.70 | 0.90 | 100.0% | 98.5% |
| no_prewarm_madvise_off | decode_plain_seed | 3 | 0 | 0 | 4.00 | 1 | 115.53 | 99.5% | 151.38 | 0.70 | 0.90 | 100.0% | 98.0% |
| no_prewarm_madvise_on | decode_json_seed | 1 | 0 | 1 | 4.00 | 1 | 100.26 | 99.2% | 150.65 | 1.10 | 1.40 | 100.0% | 90.4% |
| no_prewarm_madvise_on | decode_json_seed | 2 | 0 | 1 | 4.00 | 1 | 101.73 | 99.8% | 153.15 | 1.10 | 1.40 | 100.0% | 97.3% |
| no_prewarm_madvise_on | decode_json_seed | 3 | 0 | 1 | 4.00 | 1 | 101.75 | 99.7% | 152.89 | 1.10 | 1.30 | 100.0% | 97.4% |
| no_prewarm_madvise_on | decode_plain_seed | 1 | 0 | 1 | 4.00 | 1 | 104.25 | 99.3% | 152.35 | 1.00 | 1.10 | 100.0% | 96.6% |
| no_prewarm_madvise_on | decode_plain_seed | 2 | 0 | 1 | 4.00 | 1 | 106.79 | 99.2% | 153.04 | 0.90 | 1.10 | 100.0% | 96.4% |
| no_prewarm_madvise_on | decode_plain_seed | 3 | 0 | 1 | 4.00 | 1 | 104.01 | 99.5% | 151.28 | 0.90 | 1.10 | 100.0% | 97.6% |

## Phase Detail

| case | prompt | repeat | prefill s | prefill disk GiB | prefill CPU | decode s | decode disk GiB | decode CPU | trace rounds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no_prewarm_madvise_off | decode_json_seed | 1 |  |  |  |  |  |  |  |
| no_prewarm_madvise_off | decode_json_seed | 2 |  |  |  |  |  |  |  |
| no_prewarm_madvise_off | decode_json_seed | 3 |  |  |  |  |  |  |  |
| no_prewarm_madvise_off | decode_plain_seed | 1 |  |  |  |  |  |  |  |
| no_prewarm_madvise_off | decode_plain_seed | 2 |  |  |  |  |  |  |  |
| no_prewarm_madvise_off | decode_plain_seed | 3 |  |  |  |  |  |  |  |
| no_prewarm_madvise_on | decode_json_seed | 1 |  |  |  |  |  |  |  |
| no_prewarm_madvise_on | decode_json_seed | 2 |  |  |  |  |  |  |  |
| no_prewarm_madvise_on | decode_json_seed | 3 |  |  |  |  |  |  |  |
| no_prewarm_madvise_on | decode_plain_seed | 1 |  |  |  |  |  |  |  |
| no_prewarm_madvise_on | decode_plain_seed | 2 |  |  |  |  |  |  |  |
| no_prewarm_madvise_on | decode_plain_seed | 3 |  |  |  |  |  |  |  |
