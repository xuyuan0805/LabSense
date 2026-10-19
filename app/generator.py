from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 20260921
UNIQUE_PATIENTS = 400
DUPLICATE_ROWS = 8


def _clip(rng: np.random.Generator, mean: float, sd: float, low: float, high: float) -> float:
    return float(np.clip(rng.normal(mean, sd), low, high))


def _admission_diagnoses(rng: np.random.Generator, group: str) -> str:
    diagnoses: list[str] = []
    if group in {"thal", "iron", "chronic", "macro"} and rng.random() < .70:
        diagnoses.append("贫血待查")
    # "疑似地中海贫血"是入院时的弱线索，不允许它直接替代血常规判断。
    if group == "thal" and rng.random() < .12:
        diagnoses.append("疑似地中海贫血")
    if group != "thal" and rng.random() < .08:
        diagnoses.append("疑似地中海贫血")
    if group == "iron" and rng.random() < .31:
        diagnoses.append("缺铁待查")
    if group == "chronic" and rng.random() < .44:
        diagnoses.append("慢性肾病")
    if rng.random() < .30:
        diagnoses.append("高血压")
    if rng.random() < .22:
        diagnoses.append("糖尿病")
    if rng.random() < .18:
        diagnoses.append("感染/发热")
    if not diagnoses:
        diagnoses.append("健康体检")
    return ";".join(sorted(set(diagnoses)))


def _profile(group: str) -> dict[str, tuple[float, float]]:
    return {
        "thal": {"hgb": (109, 19), "rbc": (4.50, .62), "mcv": (76, 10.4), "mch": (24.4, 3.5), "mchc": (317, 14), "rdw": (15.2, 2.5), "wbc": (6.4, 1.9), "plt": (266, 70), "nlr": (2.15, .88)},
        "iron": {"hgb": (106, 18), "rbc": (4.35, .59), "mcv": (75, 9.0), "mch": (23.8, 3.1), "mchc": (315, 13), "rdw": (16.1, 2.5), "wbc": (6.8, 1.9), "plt": (306, 89), "nlr": (2.55, .96)},
        "chronic": {"hgb": (111, 18), "rbc": (4.08, .55), "mcv": (84, 10), "mch": (27.3, 3.3), "mchc": (323, 12), "rdw": (15.0, 2.2), "wbc": (7.5, 2.0), "plt": (238, 67), "nlr": (3.05, 1.06)},
        "macro": {"hgb": (104, 14), "rbc": (3.48, .40), "mcv": (105, 8), "mch": (33.5, 2.7), "mchc": (319, 11), "rdw": (15.2, 1.9), "wbc": (6.1, 1.3), "plt": (208, 48), "nlr": (2.35, .66)},
        "normal": {"hgb": (137, 13), "rbc": (4.55, .45), "mcv": (89, 6), "mch": (29.5, 2), "mchc": (329, 9), "rdw": (13.4, 1.25), "wbc": (6.2, 1.35), "plt": (246, 52), "nlr": (2.12, .58)},
    }[group]


def _ferritin(rng: np.random.Generator, group: str) -> float:
    """生成临床上合理、但不能单独决定标签的右偏铁蛋白值。"""
    median = {"thal": 82, "iron": 14, "chronic": 145, "macro": 92, "normal": 68}[group]
    sigma = {"thal": .72, "iron": .62, "chronic": .78, "macro": .70, "normal": .76}[group]
    value = float(rng.lognormal(np.log(median), sigma))
    # 少量急性期反应/铁负荷高值，使原始量表呈清楚的长右尾。
    if rng.random() < .035:
        value *= float(rng.uniform(3.0, 7.0))
    return float(np.clip(value, 1.2, 1800.0))


def generate_dataset() -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(SEED)
    noise_rng = np.random.default_rng(SEED + 17)
    groups = rng.choice(["thal", "iron", "chronic", "macro", "normal"], UNIQUE_PATIENTS, p=[.24, .27, .16, .08, .25])
    dates = pd.Timestamp("2026-03-01") + pd.to_timedelta(rng.integers(0, 90, UNIQUE_PATIENTS), unit="D")
    surnames = np.array(list("张李王赵陈刘杨黄周吴徐孙胡朱高林何郭马罗梁宋郑谢韩唐冯于董萧程曹袁邓许傅沈曾彭吕苏卢蒋蔡贾丁魏薛叶阎余潘杜戴夏钟汪田任姜范方石姚谭廖邹熊金陆郝孔白崔康毛邱秦江史顾侯邵孟龙万段雷钱汤尹黎易常武乔贺赖龚文"))
    discharge = {"thal": "地中海贫血", "iron": "缺铁性贫血", "chronic": "慢性病性贫血", "macro": "大细胞性贫血", "normal": "非地中海贫血"}
    records: list[dict] = []
    for i, group in enumerate(groups, 1):
        p = _profile(group); male = int(rng.integers(0, 2)); age = int(np.clip(np.rint(rng.normal(48, 16)), 18, 86))
        hgb = _clip(rng, p["hgb"][0] + 4 * male, p["hgb"][1], 28, 184); rbc = _clip(rng, p["rbc"][0] + .10 * male, p["rbc"][1], 1.8, 6.8)
        mcv = _clip(rng, p["mcv"][0], p["mcv"][1], 54, 128); mch = _clip(rng, p["mch"][0], p["mch"][1], 15, 39); mchc = _clip(rng, p["mchc"][0], p["mchc"][1], 275, 365); rdw = _clip(rng, p["rdw"][0], p["rdw"][1], 10.5, 25)
        # 两种地中海贫血表现型形成临床上常见但非线性的组合线索：
        # 低MCV伴相对高RBC，或轻度小细胞并且RDW不明显升高。
        if group == "thal" and rng.random() < .34:
            mcv = _clip(rng, 68, 5.8, 54, 95); rbc = _clip(rng, 4.98 + .10 * male, .45, 3.2, 6.8); rdw = _clip(rng, 13.7, 1.35, 10.5, 20)
        wbc = _clip(rng, p["wbc"][0], p["wbc"][1], 2, 18); plt = _clip(rng, p["plt"][0], p["plt"][1], 70, 650); lymph = _clip(rng, 1.84, .37, .55, 3.4); nlr = _clip(rng, p["nlr"][0], p["nlr"][1], .45, 8.5); neut = _clip(rng, nlr * lymph, .45, .7, 14)
        # 先按原生成顺序完成核心字段，再用独立随机流生成教学噪声指标，避免改变既有病例结构。
        patient_name = f"{rng.choice(surnames)}某{i:03d}"
        admission_diagnoses = _admission_diagnoses(rng, group)
        ferritin = _ferritin(rng, group)
        anemia_treatment = "是" if group != "normal" and rng.random() < .73 else "否"
        # 这些也是常见血常规/分类计数指标，但在本教学数据中与地贫金标准没有预设关系。
        # 它们用于真实演示：候选变量并非越多越好，训练折内筛选应排除不稳定噪声。
        mpv = _clip(noise_rng, 10.2, 1.05, 6.5, 14.5)
        pdw = _clip(noise_rng, 13.4, 2.0, 8.0, 21.0)
        monocyte = _clip(noise_rng, .46, .16, .08, 1.30)
        eosinophil = float(np.clip(noise_rng.gamma(1.7, .11), .01, 1.25))
        basophil = float(np.clip(noise_rng.gamma(1.3, .025), .005, .22))
        records.append({"record_id": f"R{i:04d}", "patient_id": f"P{i:04d}", "name": patient_name, "id_number": f"SIM-2026-{i:06d}", "sex_raw": "男" if male else "女", "age": age, "test_date": dates[i-1].strftime("%Y-%m-%d"), "admission_diagnoses": admission_diagnoses, "hgb_value": round(hgb, 1), "hgb_unit": "g/L", "wbc": round(wbc, 2), "rbc": round(rbc, 2), "mcv": round(mcv, 1), "mch": round(mch, 1), "mchc": round(mchc, 1), "rdw": round(rdw, 1), "plt": round(plt, 1), "neut": round(neut, 2), "lymph": round(lymph, 2), "mpv": round(mpv, 2), "pdw": round(pdw, 2), "monocyte": round(monocyte, 2), "eosinophil": round(eosinophil, 2), "basophil": round(basophil, 3), "ferritin_ng_ml": round(ferritin, 1), "discharge_diagnosis": discharge[group], "anemia_treatment": anemia_treatment, "note": "教学模拟数据"})
    frame = pd.DataFrame(records)
    severe_indices = rng.choice(frame.index, 6, replace=False).tolist()
    for offset, idx in enumerate(severe_indices):
        frame.loc[idx, "hgb_value"] = float(33 + offset); frame.loc[idx, "hgb_unit"] = "g/L"
    protected = set(severe_indices); candidates = [idx for idx in frame.index if idx not in protected]
    unit_indices = rng.choice(candidates, 20, replace=False).tolist()
    for idx in unit_indices:
        frame.loc[idx, "hgb_value"] = round(float(frame.loc[idx, "hgb_value"]) / 10, 1); frame.loc[idx, "hgb_unit"] = "g/dL"
    remaining = [idx for idx in candidates if idx not in unit_indices]
    # 有足够的真实缺失，学生才看得出删除病例、简单填补和邻近病例填补的取舍。
    # 缺失多发生在低HGB与边界HGB人群，模拟“复检结果未回传”的非完全随机情形。
    low_hgb = [idx for idx in remaining if float(frame.loc[idx, "hgb_value"]) < 112]
    other_hgb = [idx for idx in remaining if idx not in low_hgb]
    missing_indices = rng.choice(low_hgb, 22, replace=False).tolist() + rng.choice(other_hgb, 10, replace=False).tolist()
    frame.loc[missing_indices, "hgb_value"] = np.nan
    remaining = [idx for idx in remaining if idx not in missing_indices]
    negative_age_indices = rng.choice(remaining, 3, replace=False).tolist(); frame.loc[negative_age_indices, "age"] = [-3, -1, -7]
    male_indices = frame.index[frame.sex_raw.eq("男")].tolist(); female_indices = frame.index[frame.sex_raw.eq("女")].tolist(); male_variants = rng.choice(male_indices, 28, replace=False).tolist(); female_variants = rng.choice(female_indices, 20, replace=False).tolist()
    frame.loc[male_variants[:16], "sex_raw"] = "M"; frame.loc[male_variants[16:], "sex_raw"] = "1"; frame.loc[female_variants[:12], "sex_raw"] = "F"; frame.loc[female_variants[12:], "sex_raw"] = "0"
    duplicate_source_indices = rng.choice(frame.index, DUPLICATE_ROWS, replace=False).tolist(); frame = pd.concat([frame, frame.loc[duplicate_source_indices].copy()], ignore_index=True)
    manifest = {"seed": SEED, "unique_patients": UNIQUE_PATIENTS, "total_rows": int(len(frame)), "duplicate_rows_added": DUPLICATE_ROWS, "target": "最终出院诊断是否为地中海贫血", "ferritin_generation": "按疾病组设置重叠的对数正态分布，并加入3.5%高值；范围1.2–1800 ng/mL", "nuisance_features": {"columns":["mpv","pdw","monocyte","eosinophil","basophil"],"mechanism":"使用独立固定随机流生成，与疾病组无预设关系，用于演示训练折内筛选噪声变量"}, "injected": {"g_dl_record_ids": records_for(frame, unit_indices), "negative_age_record_ids": records_for(frame, negative_age_indices), "missing_hgb_record_ids": records_for(frame, missing_indices), "critical_hgb_record_ids": records_for(frame, severe_indices), "duplicate_source_record_ids": records_for(frame, duplicate_source_indices)}, "notice": "全部记录均为固定随机种子生成的教学模拟数据，不含真实患者信息。"}
    return frame, manifest


def records_for(frame: pd.DataFrame, indices: list[int]) -> list[str]:
    return frame.loc[indices, "record_id"].astype(str).tolist()


def write_dataset(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame, manifest = generate_dataset(); frame.to_csv(output_dir / "cbc_teaching_raw.csv", index=False, encoding="utf-8-sig")
    (output_dir / "generation_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    write_dataset(Path(__file__).resolve().parents[1] / "data")
