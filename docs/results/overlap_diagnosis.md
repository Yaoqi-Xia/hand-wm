# Block-block overlap diagnosis


## test

### main: 127 events in 300 rollouts (0.42/rollout, 34.0% of rollouts)
- onset step %: {'1-10': 3.149606299212598, '11-50': 29.92125984251969, '51-100': 31.496062992125985, '101-200': 35.43307086614173}  (median 75.0)
- max depth %: {'5-10mm': 19.68503937007874, '10-20mm': 18.11023622047244, '20-40mm': 43.30708661417323, '>40mm': 18.89763779527559}
- category %: {'push_chain': 81.10236220472441, 'resting_sinking': 1.5748031496062993, 'drift': 4.724409448818897, 'other': 12.598425196850394}
- push chain: {'n': 103, 'pushed_pred_minus_true_cm_median': 0.0, 'hit_pred_minus_true_cm_median': -0.07996296635555092, 'pct_hit_moves_too_little(>0.5cm short)': 36.89320388349515, 'pct_pushed_moves_too_far(>0.5cm over)': 24.271844660194176, 'pct_both': 3.8834951456310676}
- by initial nearest-neighbour distance: {'1 block': {'rollouts': 48, 'events_per_rollout': 0.0}, '10-15cm': {'rollouts': 63, 'events_per_rollout': 0.31746031746031744}, '7-10cm': {'rollouts': 120, 'events_per_rollout': 0.675}, '<7cm': {'rollouts': 26, 'events_per_rollout': 1.0}, '>15cm': {'rollouts': 43, 'events_per_rollout': 0.0}}
### action: 53 events in 300 rollouts (0.18/rollout, 15.7% of rollouts)
- onset step %: {'1-10': 1.8867924528301887, '11-50': 32.075471698113205, '51-100': 20.754716981132077, '101-200': 45.28301886792453}  (median 83.0)
- max depth %: {'5-10mm': 20.754716981132077, '10-20mm': 32.075471698113205, '20-40mm': 37.735849056603776, '>40mm': 9.433962264150944}
- category %: {'push_chain': 54.716981132075475, 'resting_sinking': 7.547169811320755, 'drift': 13.20754716981132, 'other': 24.528301886792452}
- push chain: {'n': 29, 'pushed_pred_minus_true_cm_median': -0.9209601018454889, 'hit_pred_minus_true_cm_median': 0.0, 'pct_hit_moves_too_little(>0.5cm short)': 27.586206896551722, 'pct_pushed_moves_too_far(>0.5cm over)': 6.896551724137931, 'pct_both': 0.0}
- by initial nearest-neighbour distance: {'1 block': {'rollouts': 48, 'events_per_rollout': 0.0}, '10-15cm': {'rollouts': 63, 'events_per_rollout': 0.09523809523809523}, '7-10cm': {'rollouts': 120, 'events_per_rollout': 0.24166666666666667}, '<7cm': {'rollouts': 26, 'events_per_rollout': 0.6538461538461539}, '>15cm': {'rollouts': 43, 'events_per_rollout': 0.023255813953488372}}
### mlp: 18 events in 300 rollouts (0.06/rollout, 5.3% of rollouts)
- onset step %: {'1-10': 0.0, '11-50': 5.555555555555555, '51-100': 22.22222222222222, '101-200': 72.22222222222221}  (median 124.0)
- max depth %: {'5-10mm': 44.44444444444444, '10-20mm': 55.55555555555556, '20-40mm': 0.0, '>40mm': 0.0}
- category %: {'push_chain': 22.22222222222222, 'resting_sinking': 11.11111111111111, 'drift': 27.77777777777778, 'other': 38.888888888888886}
- push chain: {'n': 4, 'pushed_pred_minus_true_cm_median': -3.6348052083242837, 'hit_pred_minus_true_cm_median': -0.02231562191835941, 'pct_hit_moves_too_little(>0.5cm short)': 25.0, 'pct_pushed_moves_too_far(>0.5cm over)': 0.0, 'pct_both': 0.0}
- by initial nearest-neighbour distance: {'1 block': {'rollouts': 48, 'events_per_rollout': 0.0}, '10-15cm': {'rollouts': 63, 'events_per_rollout': 0.0}, '7-10cm': {'rollouts': 120, 'events_per_rollout': 0.03333333333333333}, '<7cm': {'rollouts': 26, 'events_per_rollout': 0.5384615384615384}, '>15cm': {'rollouts': 43, 'events_per_rollout': 0.0}}

## test_ood

### main: 664 events in 300 rollouts (2.21/rollout, 92.7% of rollouts)
- onset step %: {'1-10': 7.078313253012049, '11-50': 28.162650602409638, '51-100': 26.054216867469883, '101-200': 38.704819277108435}  (median 76.5)
- max depth %: {'5-10mm': 18.22289156626506, '10-20mm': 29.518072289156628, '20-40mm': 38.55421686746988, '>40mm': 13.704819277108435}
- category %: {'push_chain': 71.98795180722891, 'resting_sinking': 2.4096385542168677, 'drift': 8.283132530120483, 'other': 17.319277108433734}
- push chain: {'n': 478, 'pushed_pred_minus_true_cm_median': 0.0, 'hit_pred_minus_true_cm_median': -0.305765398691918, 'pct_hit_moves_too_little(>0.5cm short)': 41.63179916317992, 'pct_pushed_moves_too_far(>0.5cm over)': 26.15062761506276, 'pct_both': 5.230125523012552}
- by initial nearest-neighbour distance: {'10-15cm': {'rollouts': 1, 'events_per_rollout': 2.0}, '7-10cm': {'rollouts': 203, 'events_per_rollout': 2.167487684729064}, '<7cm': {'rollouts': 96, 'events_per_rollout': 2.3125}}
### action: 441 events in 300 rollouts (1.47/rollout, 81.0% of rollouts)
- onset step %: {'1-10': 4.761904761904762, '11-50': 26.984126984126984, '51-100': 28.34467120181406, '101-200': 39.909297052154194}  (median 81.0)
- max depth %: {'5-10mm': 24.489795918367346, '10-20mm': 34.69387755102041, '20-40mm': 34.24036281179138, '>40mm': 6.575963718820861}
- category %: {'push_chain': 49.43310657596372, 'resting_sinking': 7.709750566893424, 'drift': 11.791383219954648, 'other': 31.065759637188208}
- push chain: {'n': 218, 'pushed_pred_minus_true_cm_median': -0.18562984824800677, 'hit_pred_minus_true_cm_median': -0.028319477937666038, 'pct_hit_moves_too_little(>0.5cm short)': 35.321100917431195, 'pct_pushed_moves_too_far(>0.5cm over)': 20.18348623853211, 'pct_both': 2.7522935779816518}
- by initial nearest-neighbour distance: {'10-15cm': {'rollouts': 1, 'events_per_rollout': 0.0}, '7-10cm': {'rollouts': 203, 'events_per_rollout': 1.3694581280788178}, '<7cm': {'rollouts': 96, 'events_per_rollout': 1.6979166666666667}}
### mlp: 500 events in 300 rollouts (1.67/rollout, 80.7% of rollouts)
- onset step %: {'1-10': 10.8, '11-50': 51.800000000000004, '51-100': 18.2, '101-200': 19.2}  (median 32.5)
- max depth %: {'5-10mm': 14.399999999999999, '10-20mm': 30.599999999999998, '20-40mm': 38.0, '>40mm': 17.0}
- category %: {'push_chain': 21.4, 'resting_sinking': 4.8, 'drift': 54.0, 'other': 19.8}
- push chain: {'n': 107, 'pushed_pred_minus_true_cm_median': -0.915190394247053, 'hit_pred_minus_true_cm_median': 0.3405588799293099, 'pct_hit_moves_too_little(>0.5cm short)': 21.49532710280374, 'pct_pushed_moves_too_far(>0.5cm over)': 13.084112149532709, 'pct_both': 0.0}
- by initial nearest-neighbour distance: {'10-15cm': {'rollouts': 1, 'events_per_rollout': 2.0}, '7-10cm': {'rollouts': 203, 'events_per_rollout': 1.541871921182266}, '<7cm': {'rollouts': 96, 'events_per_rollout': 1.9270833333333333}}

## one-step check (main model, from true states)
{'test': {'steps': 38400, 'pct_steps_with_overlap': 0.3046875, 'collision_frames': 4849, 'pct_collision_frames_with_overlap': 2.103526500309342}, 'test_ood': {'steps': 38400, 'pct_steps_with_overlap': 1.3020833333333333, 'collision_frames': 21293, 'pct_collision_frames_with_overlap': 2.2072981731085335}}