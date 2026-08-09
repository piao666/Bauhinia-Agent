from __future__ import annotations

import pytest

from bauhinia_agent.self_model import ProfileSelector, SelfModelError, TaskClassification


def test_classification_normalizes_dimensions_and_selector_key_is_stable() -> None:
    classification = TaskClassification(
        project_id="project_a",
        model_config_hash="a" * 64,
        evaluator_version="eval-v1",
        environment_hash="b" * 64,
        language="Python",
        repository_scale="medium",
        task_type="BugFix",
        tool_category="PyTest",
        risk_level="low",
    )
    selector = ProfileSelector(**classification.to_dict(), verification_level="strong")

    assert classification.language == "python"
    assert classification.task_type == "bugfix"
    assert selector.key == ProfileSelector(**classification.to_dict(), verification_level="strong").key
    assert selector.dimension == "language+repository_scale+task_type+tool_category+risk_level+verification_level"


def test_project_model_evaluator_and_environment_scope_cannot_be_implicit() -> None:
    with pytest.raises(SelfModelError, match="model_config_hash"):
        ProfileSelector(
            project_id="project_a",
            model_config_hash="not-a-hash",
            evaluator_version="eval-v1",
            environment_hash="b" * 64,
        )

    with pytest.raises(SelfModelError, match="stable lowercase token"):
        TaskClassification(
            project_id="project_a",
            model_config_hash="a" * 64,
            evaluator_version="eval-v1",
            environment_hash="b" * 64,
            language="python script",
            repository_scale="small",
            task_type="bugfix",
            tool_category="pytest",
            risk_level="low",
        )
