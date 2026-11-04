# samp830 LIBERO samples

These four episodes are a format sample from
[`samp830/libero_40_suboptimal_v1`](https://huggingface.co/datasets/samp830/libero_40_suboptimal_v1).
They are one episode per LIBERO suite: two successful and two failed rollouts.
They are included so that the canonical converter and tests have a small real-data
fixture; they are **not** the full dataset.

| File | Suite | Task | Episode | raw_demo | Outcome | Length |
| --- | --- | --- | ---: | --- | --- | ---: |
| `libero_10_episode_000232.parquet` | LIBERO-10 | pick up the book and place it in the back compartment of the caddy | 232 | demo_7 | success | 147 |
| `libero_goal_episode_000488.parquet` | LIBERO-Goal | open the middle drawer of the cabinet | 488 | demo_13 | failure | 500 |
| `libero_object_episode_000672.parquet` | LIBERO-Object | pick up the salad dressing and place it in the basket | 672 | demo_22 | success | 110 |
| `libero_spatial_episode_000847.parquet` | LIBERO-Spatial | pick up the black bowl on the stove and place it on the plate | 847 | demo_22 | failure | 500 |

Source revision: `71a88ad0eaa5f5dc5237e6ede56e7cc3d4a3e9e8`.

## Parquet fields

Each file is one episode and contains:

| Field | Shape / type | Meaning |
| --- | --- | --- |
| `image` | 256x256 PNG bytes | external camera |
| `wrist_image` | 256x256 PNG bytes | wrist camera |
| `state` | 8 float32 | 6-D end-effector pose + 2-D gripper |
| `actions` | 7 float32 | native LIBERO delta action, range [-1, 1] |
| `timestamp` | float | seconds, 10 fps |
| `frame_index` | int | zero-based frame index |
| `episode_index` | int | source episode id |
| `index` | int | global frame index |
| `task_index` | int | LIBERO task id |

The parquet files do not contain `reward`, `done`, or `success`.
Success and length are in `manifest.jsonl` and in the source
`episode_manifest.jsonl`.

## Mapping to the canonical schema

For these samples, the intended mapping is:

- `obs = state`
- `action = actions`
- `reward = 1.0` only on the final transition of a successful episode, otherwise `0.0`
- `terminated = True` on the final transition (success or task-horizon exhaustion)
- `truncated = False`
- `task_horizon = 500`
- `remaining_steps = 500 - frame_index`
- `policy_id = "samp830/flow_bc/<suite>"`
- `task_id = "<suite>/task_<task_index>"`

Failure episodes are 500-step timeouts, so the final transition is terminal by
task-horizon exhaustion, not collector truncation.

## Provenance and license

The source dataset card does not declare a license. These four files are included
only as a small format/testing fixture. Do not redistribute the full dataset from
this repository; use the Hugging Face dataset and cite its source. The `raw_demo`
field is the source demonstration index and does not necessarily align with
`demo_i` episode ordering in other LIBERO conversions.

## Full data

The full dataset is 1000 episodes, 40 tasks, about 28 GB. It is not stored in this
repository. Use the Hugging Face link above, pin the revision, and verify against
`samples/checksums.sha256` for the four files here.
