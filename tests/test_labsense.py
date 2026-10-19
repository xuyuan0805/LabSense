from pathlib import Path

import pandas.testing as pdt
import pytest

from app.engine import ClassroomEngine, ISSUE_ORDER
from app.generator import DUPLICATE_ROWS, UNIQUE_PATIENTS, generate_dataset


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "cbc_teaching_raw.csv"
ANSWERS = {"unit":"convert","age":"missing","sex":"map","duplicate":"dedupe","critical":"flag_keep","privacy":"remove","missing_hgb":"experiment"}


def prepared() -> ClassroomEngine:
    engine = ClassroomEngine(DATA)
    engine.run_checks()
    for issue in ISSUE_ORDER:
        assert engine.answer_decision(issue, ANSWERS[issue])["is_correct"]
    return engine


def test_fixed_generator_has_required_admission_and_gold_standard_fields():
    first, _ = generate_dataset(); second, _ = generate_dataset()
    pdt.assert_frame_equal(first, second)
    assert len(first) == UNIQUE_PATIENTS + DUPLICATE_ROWS
    assert {"admission_diagnoses", "discharge_diagnosis", "ferritin_ng_ml"}.issubset(first.columns)
    assert first.ferritin_ng_ml.skew() > 1
    assert first["discharge_diagnosis"].eq("地中海贫血").any()
    assert set(first.sex_raw) >= {"男", "女", "M", "F", "1", "0"}


def test_only_standard_answer_advances_and_builds_model_frame():
    engine = ClassroomEngine(DATA); engine.run_checks()
    wrong = engine.answer_decision("unit", "ignore")
    assert wrong["is_correct"] is False and engine.state()["answered"] == 0
    engine = prepared()
    assert engine.model_frame is not None
    assert set(engine.model_frame.sex_binary.dropna()) <= {0.0, 1.0}
    assert {"name", "id_number"}.isdisjoint(engine.model_frame.columns)
    assert engine.model_frame.hgb_gl.lt(40).any()


def test_feature_preview_keeps_gold_standard_out_of_inputs():
    engine = prepared(); preview = engine.feature_preview()
    assert preview["admission_encoding"]["columns"]
    assert preview["sex_mapping"][0]["模型数值"] == 1
    assert "thalassemia_label（训练标签）" in preview["excluded"]
    assert preview["processing_catalog"]
    assert preview["ferritin_transform"]["raw"]["summary"]["skewness"] > preview["ferritin_transform"]["log"]["summary"]["skewness"]
    assert len(preview["ferritin_transform"]["model_comparison"]) == 2


def test_imputation_requires_training_fold_comparison():
    engine = prepared(); result = engine.run_imputation()
    assert [x["method"] for x in result["methods"]] == ["complete", "median", "knn"]
    assert result["recommended"] in {"median", "knn"}
    assert all(0 <= x["metrics"]["roc_auc"] <= 1 for x in result["methods"])
    assert result["observed_summary"]["count"] + result["distributions"][0]["filled_summary"]["count"] == len(engine.model_frame)
    assert result["distributions"][0]["completed_summary"]["count"] == len(engine.model_frame)
    assert result["auc_explanation"]["missing_records"] == 32
    assert len(result["auc_explanation"]["fold_rows"]) == 5
    assert result["auc_explanation"]["auc_gap"] > 0
    assert result["auc_explanation"]["same_at_three_decimals"] is True


def test_lasso_selection_is_fold_safe_and_reproducible():
    engine = prepared(); engine.run_imputation(); result = engine.run_feature_selection()
    assert result["method"] == "相关性预筛 + 标准LASSO（L1逻辑回归）"
    assert len(result["lambda_sequence"]) == 45
    assert len(result["coefficient_path"]["coefficients"]) == 45
    assert result["lambda_1se"] >= result["lambda_min"]
    assert len(result["selected_1se"]) <= len(result["selected_min"])
    assert all(row["patient_overlap"] == 0 for row in engine._evaluate(engine.model_frame, "logistic", engine.imputation_result["recommended"], result["recommended_c"])["folds"])
    assert result["frequency"]
    assert all(row["patient_overlap"] == 0 for row in result["folds"])
    assert any("NLR" in row["removed"] for row in result["correlation_filter"])


def test_lasso_comparison_improves_validation_and_reduces_gap():
    engine = prepared(); engine.run_imputation(); result = engine.run_feature_selection(); comparison = engine.run_compare()
    assert result["recommended_lambda"] in {"lambda_min", "lambda_1se"}
    assert len(comparison["models"]) == 8
    logistic_rows = [row for row in comparison["models"] if row["key"] in {"logistic", "lasso"}]
    assert {row["feature_set"] for row in logistic_rows} == {"全部特征", "LASSO筛选后"}
    assert all(fold["patient_overlap"] == 0 for row in logistic_rows for fold in row["folds"])
    for key in {"tree", "forest", "svm"}:
        rows = [row for row in comparison["models"] if row["key"] == key]
        assert {row["feature_set"] for row in rows} == {"全部特征", "LASSO筛选后"}
        assert all(fold["patient_overlap"] == 0 for row in rows for fold in row["folds"])
    assert len(comparison["pairwise_effects"]) == 4
    assert comparison["selection_effect"]["improved_auc"] is True
    assert comparison["selection_effect"]["reduced_gap"] is True
    assert comparison["selection_effect"]["auc_change"] > 0
    assert comparison["selection_effect"]["gap_change"] < 0
    assert comparison["winner"]["key"] == "lasso"
