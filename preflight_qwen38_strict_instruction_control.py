#!/usr/bin/env python3
"""Validate the six-dataset strict semantic instruction control."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import evaluate_qwen38_orion as base
import evaluate_qwen38_recipe as recipe
from preflight_qwen38_instruction_study import SUBSET_DATASETS

EXPECTED = (
    recipe.Condition(
        "strict_permuted_instructions_multi",
        "multi",
        "class_names",
        "none",
        0,
        instruction_mode="strict_permuted",
    ),
)


def validate(dataset_root: Path, conditions_path: Path) -> dict:
    conditions = recipe.load_conditions(conditions_path)
    if conditions != EXPECTED:
        raise ValueError("Strict-control condition differs from the locked contract.")
    rows = []
    for name in SUBSET_DATASETS:
        dataset = dataset_root / name
        test_directory = dataset / "test"
        test = base.load_coco(test_directory / "_annotations.coco.json")
        categories = base.categories_by_id(test)
        readme_path = dataset / "README.dataset.txt"
        readme = readme_path.read_text(encoding="utf-8").strip()
        detailed_only = recipe.permute_class_instruction_sections(readme)
        strict = recipe.permute_all_class_guidance(readme)
        if strict == detailed_only or strict == readme:
            raise ValueError(f"Strict control did not alter both guidance levels for {name}.")
        if sorted(re.findall(r"\S+", strict)) != sorted(re.findall(r"\S+", readme)):
            raise ValueError(f"Strict control changed README vocabulary for {name}.")
        image = min(test["images"], key=lambda value: int(value["id"]))
        task = base.Task(
            mode=conditions[0].mode,
            image_id=int(image["id"]),
            file_name=str(image["file_name"]),
            width=int(image["width"]),
            height=int(image["height"]),
        )
        content = recipe.build_messages(
            task,
            conditions[0],
            test_directory,
            categories,
            {},
            {},
            {},
            readme,
        )[0]["content"]
        if strict not in content[0]["text"] or readme in content[0]["text"]:
            raise ValueError(f"Strict control prompt mismatch for {name}.")
        if content[-2] != {"type": "text", "text": "TARGET IMAGE:"}:
            raise ValueError(f"Target marker mismatch for {name}.")
        rows.append(
            {
                "dataset": name,
                "test_images": len(test["images"]),
                "classes": len(categories),
                "readme_sha256": base.sha256_file(readme_path),
            }
        )
    return {
        "created_at": base.utc_now(),
        "dataset_root": str(dataset_root),
        "condition_sha256": base.sha256_file(conditions_path),
        "dataset_count": len(rows),
        "test_images": sum(row["test_images"] for row in rows),
        "request_count": sum(row["test_images"] for row in rows),
        "temperature": 0,
        "seed": 1234,
        "reasoning": "off",
        "same_readme_vocabulary": True,
        "introduction_definitions_permuted": True,
        "detailed_class_sections_permuted": True,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.dataset_root.resolve(), args.conditions.resolve())
    base.atomic_write_json(args.report.resolve(), result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
