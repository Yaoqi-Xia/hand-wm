
## test
(a) one-step error from true state, mm. share of block-steps: {'static': 80.27, 'moving': 1.84, 'contacted': 17.89}
| method | static | moving | contacted |
|---|---|---|---|
| main | 0.18 | 4.37 | 1.80 |
| main_onestep | 0.10 | 4.15 | 1.78 |
| action | 1.26 | 4.96 | 2.88 |
| mlp | 0.32 | 5.15 | 2.07 |
| static | 0.00 | 6.50 | 3.00 |
| const_velocity | 0.03 | 6.33 | 2.07 |
(b) rollout error, cm. shares: {'@10': {'never_moved': 80.69, 'moved_not_hand_touched': 1.25, 'hand_touched': 18.06}, '@50': {'never_moved': 43.95, 'moved_not_hand_touched': 4.18, 'hand_touched': 51.88}, '@200': {'never_moved': 9.39, 'moved_not_hand_touched': 5.85, 'hand_touched': 84.76}}
| method | never_moved @10 | moved_not_hand_touched @10 | hand_touched @10 | never_moved @50 | moved_not_hand_touched @50 | hand_touched @50 | never_moved @200 | moved_not_hand_touched @200 | hand_touched @200 |
|---|---|---|---|---|---|---|---|---|---|
| main | 0.07 | 1.23 | 0.97 | 0.22 | 1.72 | 2.24 | 0.68 | 2.27 | 4.46 |
| main_onestep | 0.05 | 1.13 | 1.04 | 0.22 | 1.92 | 2.71 | 0.88 | 2.30 | 5.37 |
| action | 0.23 | 0.80 | 1.44 | 0.47 | 1.74 | 3.24 | 1.12 | 2.32 | 5.97 |
| mlp | 0.25 | 0.76 | 1.37 | 0.86 | 1.69 | 3.61 | 1.84 | 2.44 | 6.32 |
| static | 0.00 | 0.84 | 1.55 | 0.00 | 1.84 | 4.04 | 0.00 | 2.56 | 6.62 |
| const_velocity | 0.01 | 3.25 | 2.12 | 0.03 | 3.55 | 5.91 | 0.00 | 4.44 | 12.22 |

## test_ood
(a) one-step error from true state, mm. share of block-steps: {'static': 81.98, 'moving': 1.28, 'contacted': 16.74}
| method | static | moving | contacted |
|---|---|---|---|
| main | 0.20 | 4.54 | 1.88 |
| main_onestep | 0.10 | 4.26 | 1.86 |
| action | 1.19 | 4.97 | 2.82 |
| mlp | 0.89 | 5.82 | 2.62 |
| static | 0.00 | 6.40 | 2.85 |
| const_velocity | 0.02 | 6.43 | 2.06 |
(b) rollout error, cm. shares: {'@10': {'never_moved': 82.18, 'moved_not_hand_touched': 3.74, 'hand_touched': 14.08}, '@50': {'never_moved': 49.19, 'moved_not_hand_touched': 10.25, 'hand_touched': 40.57}, '@200': {'never_moved': 10.78, 'moved_not_hand_touched': 16.14, 'hand_touched': 73.08}}
| method | never_moved @10 | moved_not_hand_touched @10 | hand_touched @10 | never_moved @50 | moved_not_hand_touched @50 | hand_touched @50 | never_moved @200 | moved_not_hand_touched @200 | hand_touched @200 |
|---|---|---|---|---|---|---|---|---|---|
| main | 0.08 | 1.17 | 1.23 | 0.22 | 1.88 | 2.42 | 0.60 | 2.53 | 4.28 |
| main_onestep | 0.05 | 1.26 | 1.42 | 0.23 | 1.94 | 2.94 | 0.86 | 2.76 | 4.90 |
| action | 0.19 | 1.27 | 1.63 | 0.44 | 2.02 | 3.35 | 1.07 | 2.99 | 5.59 |
| mlp | 0.94 | 1.95 | 2.01 | 4.78 | 6.04 | 5.97 | 19.89 | 22.43 | 20.19 |
| static | 0.00 | 1.41 | 1.83 | 0.00 | 2.25 | 3.90 | 0.00 | 3.06 | 5.86 |
| const_velocity | 0.01 | 2.52 | 2.43 | 0.04 | 4.01 | 6.58 | 0.14 | 4.96 | 13.61 |