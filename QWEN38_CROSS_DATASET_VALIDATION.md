# Qwen3.8-Max cross-dataset prompt validation

Status: prepared

This validation tests the six selected Orion recipes on two additional
RF20-VL-FSOD test sets:

| Dataset | Domain | Test images | Classes | Ground-truth objects |
|---|---|---:|---:|---:|
| `lacrosse-object-detection` | Sports | 50 | 4 | 355 |
| `the-dreidel-project` | Fine-grained objects and symbols | 54 | 6 | 171 |

Both test splits contain ground truth for every declared class. References are
selected deterministically from each dataset's train split. Test images and
class names are the only target-side information supplied to the model.

The evaluated recipes are multi-class class names, per-class positive numeric
boxes, per-class positive drawn boxes, per-class positive and negative numeric
boxes, and one-call multi-class positive numeric boxes with reasoning `none`
and `low`. Per-class predictions are merged before COCO scoring. All metrics use
pycocotools with `maxDets=[1, 10, 500]`.

The complete matrix requires 750 API calls for lacrosse and 1,134 for dreidel.
Every response is atomically checkpointed, and the launcher retries resumable
invocations without repeating terminal fingerprint-matching records.

Run both datasets with:

```bash
bash run_qwen38_cross_dataset_validation.sh
```
