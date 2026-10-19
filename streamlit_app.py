from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from app.engine import ClassroomEngine, FIELD_LABELS, ISSUE_ORDER


ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="LabSense 地中海贫血教学实验室", page_icon="🩸", layout="wide")


@st.cache_data(show_spinner=False)
def dataframe_from_records(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def engine() -> ClassroomEngine:
    if "classroom_engine" not in st.session_state:
        st.session_state.classroom_engine = ClassroomEngine(ROOT / "data" / "cbc_teaching_raw.csv")
    return st.session_state.classroom_engine


def title(number: int, name: str, subtitle: str) -> None:
    st.caption(f"第 {number:02d} 章 · 在线教学实验")
    st.title(name)
    st.markdown(f"<p class='chapter-subtitle'>{subtitle}</p>", unsafe_allow_html=True)


def metric_cards(items: list[tuple[str, object, str | None]]) -> None:
    cols = st.columns(len(items))
    for col, (label, value, help_text) in zip(cols, items):
        col.metric(label, value, help=help_text)


def progress_runner(label: str, fn):
    bar = st.progress(0, text=f"{label}：准备数据")

    def update(value: int, message: str) -> None:
        bar.progress(min(max(int(value), 0), 100), text=message)

    try:
        result = fn(update)
        bar.progress(100, text=f"{label}：计算完成")
        return result
    except Exception as exc:
        bar.empty()
        st.error(str(exc))
        return None


def require(condition: bool, message: str) -> bool:
    if condition:
        return True
    st.info(message)
    return False


def hist_frame(items: list[dict], series: str) -> pd.DataFrame:
    return pd.DataFrame({"区间": [f"{x['from']}–{x['to']}" for x in items], series: [x["count"] for x in items]})


st.markdown(
    """
<style>
    .stApp { background: linear-gradient(180deg,#f4faf9 0,#ffffff 340px); }
    .block-container { max-width: 1320px; padding-top: 2rem; padding-bottom: 4rem; }
    h1,h2,h3 { color:#073b4c; letter-spacing:-.02em; }
    .chapter-subtitle { color:#52717c; font-size:1.06rem; margin-top:-.7rem; }
    [data-testid="stMetric"] { background:white; border:1px solid #d9e8e7; border-radius:16px; padding:16px; }
    div[data-testid="stDataFrame"] { border:1px solid #d9e8e7; border-radius:14px; overflow:hidden; }
    .lesson-card { background:white; border:1px solid #d9e8e7; border-radius:18px; padding:20px; margin:8px 0 18px; box-shadow:0 8px 28px rgba(5,70,80,.05); }
    .flow { display:flex; gap:10px; flex-wrap:wrap; align-items:center; margin:18px 0; }
    .flow span { background:#e2f4f1; color:#006d66; border-radius:999px; padding:11px 17px; font-weight:700; }
    .flow b { color:#0b8f83; font-size:1.3rem; }
</style>
""",
    unsafe_allow_html=True,
)

eng = engine()
chapters = [
    "研究任务", "查看原始数据", "数据质量扫描", "数据处理", "缺失值实验",
    "特征工程", "相关性与 LASSO", "模型比较", "SHAP 解释", "PCA 与聚类",
]
with st.sidebar:
    st.markdown("## 🩸 LabSense")
    st.caption("地中海贫血提示模型 · 在线教学版")
    scene = st.radio("课程目录", range(1, 11), format_func=lambda i: f"{i:02d}  {chapters[i-1]}")
    st.divider()
    state = eng.state()
    st.caption(f"当前进度：已处理 {state['answered']}/{state['required']} 项数据问题")
    if st.button("重新开始本次实验", use_container_width=True):
        eng.reset()
        for key in list(st.session_state):
            if key.startswith("answer_"):
                del st.session_state[key]
        st.rerun()
    st.caption("固定模拟数据 · 不含真实患者信息")


if scene == 1:
    title(1, "入院血常规，能否提示地中海贫血风险？", "从历史资料出发，完成一次可复现的医学机器学习研究。")
    st.markdown(
        """<div class="flow"><span>患者入院</span><b>→</b><span>血常规 + 入院诊断</span><b>→</b><span>风险提示概率</span><b>→</b><span>住院检查</span><b>→</b><span>出院诊断金标准</span></div>""",
        unsafe_allow_html=True,
    )
    metric_cards([
        ("预测时点", "患者入院后", "模型只能使用预测当时已经获得的信息。"),
        ("模型输入", "血常规 + 入院诊断", "姓名、身份证和住院后信息禁止入模。"),
        ("模型输出", "地贫提示概率", "用于辅助提醒与科研探索，不替代诊断。"),
        ("历史金标准", "最终出院诊断", "只用于训练和验证，在真实入院预测时尚不可获得。"),
    ])
    st.warning("使用边界：这是固定模拟数据上的教学模型，不可直接用于患者诊疗、危急值处置或临床决策。")

elif scene == 2:
    title(2, "查看原始数据", "先看资料本身，不提前泄露后续扫描答案。")
    raw = eng.raw_page(1, 20)
    metric_cards([("记录数", raw["summary"]["rows"], "含重复导入记录"), ("患者数", raw["summary"]["patients"], None), ("字段数", raw["summary"]["columns"], None), ("数据来源", raw["summary"]["source"], "仓库内固定 CSV")])
    df = dataframe_from_records(raw["rows"]).rename(columns=FIELD_LABELS)
    st.dataframe(df, use_container_width=True, height=520, hide_index=True)
    with st.expander("字段角色怎么判断？"):
        st.markdown("**入院时可获得**：血常规、年龄、性别、入院诊断。  \n**训练标签**：最终出院诊断是否明确地中海贫血。  \n**禁止入模**：姓名、身份证、治疗信息、出院诊断及金标准标签。")

elif scene == 3:
    title(3, "数据质量扫描", "让规则真正扫描当前 CSV，再用图表查看证据。")
    if st.button("运行七项数据检查", type="primary"):
        eng.run_checks()
    if require(bool(eng.issues), "点击上方按钮后，系统会检查单位、年龄、编码、重复、危急值、隐私和缺失值。"):
        issues = [eng.issues[x] for x in ISSUE_ORDER]
        chart = pd.DataFrame({"问题": [x["title"] for x in issues], "命中记录": [x["affected_count"] for x in issues]}).set_index("问题")
        st.bar_chart(chart, horizontal=True, color="#0b8f83")
        cols = st.columns(2)
        with cols[0]:
            st.subheader("HGB 单位构成", help="同一指标出现 g/L 与 g/dL，直接比较数值会产生10倍误差。")
            st.bar_chart(pd.DataFrame(eng.check_distributions["units"]).set_index("label"), color="#e68a21")
        with cols[1]:
            st.subheader("性别原始编码", help="男、M、1表达同一类别，必须依据数据字典统一。")
            st.bar_chart(pd.DataFrame(eng.check_distributions["sex"]).set_index("label"), color="#3178c6")
        for issue in issues:
            with st.expander(f"{issue['title']} · 命中 {issue['affected_count']} 条"):
                st.write(f"**可能原因：** {issue['cause']}  **建模风险：** {issue['risk']}")
                st.dataframe(dataframe_from_records(issue["sample_records"]).rename(columns=FIELD_LABELS), hide_index=True, use_container_width=True)

elif scene == 4:
    title(4, "数据处理", "每一题只有一个符合本研究设计的答案；答错会解释原因，答对才进入下一项。")
    if not eng.issues:
        st.info("请先在第3章运行数据质量扫描。")
    else:
        next_issue = next((x for x in ISSUE_ORDER if x not in eng.answers), None)
        if next_issue is None:
            st.success("七项处理均已完成，分析数据已生成。")
            st.dataframe(eng.model_frame.head(12).rename(columns=FIELD_LABELS), hide_index=True, use_container_width=True)
        else:
            issue = eng.issues[next_issue]
            st.progress(len(eng.answers) / len(ISSUE_ORDER), text=f"进度 {len(eng.answers)}/{len(ISSUE_ORDER)}")
            st.subheader(issue["title"], help=f"可能原因：{issue['cause']} 建模风险：{issue['risk']}")
            st.dataframe(dataframe_from_records(issue["sample_records"]).rename(columns=FIELD_LABELS), hide_index=True, use_container_width=True)
            choices = {label: key for key, label in issue["options"]}
            selected = st.radio("本次研究应该怎么处理？", list(choices), index=None)
            if st.button("提交答案", type="primary", disabled=selected is None):
                result = eng.answer_decision(next_issue, choices[selected])
                if result["is_correct"]:
                    st.success(result["feedback"])
                    st.rerun()
                else:
                    st.error(result["feedback"])
                    st.info(result["hint"])

elif scene == 5:
    title(5, "缺失值实验", "比较删除、中位数填补和 KNN 填补：既看模型AUC，也看填补后分布。")
    if require(eng.model_frame is not None, "请先完成第4章的七项数据处理。"):
        if st.button("运行缺失值实验", type="primary"):
            progress_runner("患者级五折交叉验证", eng.run_imputation)
        if eng.imputation_result:
            result = eng.imputation_result
            scores = pd.DataFrame([{"方法": x["label"], "ROC AUC": x["metrics"]["roc_auc"], "折间标准差": x["fold_sd"], "样本数": x["sample_count"]} for x in result["methods"]])
            st.dataframe(scores, hide_index=True, use_container_width=True)
            st.bar_chart(scores.set_index("方法")[["ROC AUC"]], color="#12a179")
            st.success(f"后续管道采用：{result['recommended_label']}。{result['rule']}")
            st.subheader("HGB 填补前后分布", help="填补前只统计原来有结果的记录；填补后会增加原来缺失的记录。")
            observed = hist_frame(result["observed_hgb_histogram"], "填补前已观测")
            method = st.selectbox("查看填补方法", result["distributions"], format_func=lambda x: x["label"])
            after = hist_frame(method["histogram"], "填补后全体")
            st.bar_chart(observed.merge(after, on="区间", how="outer").fillna(0).set_index("区间"))
            with st.expander("AUC 是怎样算出来的？"):
                st.write(result["auc_explanation"]["calculation"])
                st.write(result["auc_explanation"]["why_close"])

elif scene == 6:
    title(6, "变量怎样变成特征", "根据变量类型和算法需要选择处理方法，不是所有数值都要做同一种转换。")
    if require(eng.model_frame is not None, "请先完成第4章的数据处理。"):
        preview = eng.feature_preview()
        catalog = pd.DataFrame(preview["processing_catalog"]).rename(columns={"purpose":"目的","when":"适用场景","common":"常用方法","used":"本课采用","active":"采用"})
        st.dataframe(catalog[["目的","适用场景","常用方法","本课采用"]], hide_index=True, use_container_width=True)
        st.subheader("铁蛋白为什么取 log(1+x)？", help="右偏变量有大量低值和少量极高值。对数转换压缩长右尾，减弱极端值对线性系数的影响，但不保证AUC提高。")
        ferritin = preview["ferritin_transform"]
        c1, c2 = st.columns(2)
        c1.bar_chart(hist_frame(ferritin["raw"]["histogram"], "原始铁蛋白").set_index("区间"), color="#e68a21")
        c2.bar_chart(hist_frame(ferritin["log"]["histogram"], "log(1+x)").set_index("区间"), color="#0b8f83")
        metric_cards([("原始偏度", ferritin["raw"]["summary"]["skewness"], "越大表示右尾越长"), ("转换后偏度", ferritin["log"]["summary"]["skewness"], None), ("性别编码", "男=1，女=0", "二分类直接映射"), ("连续变量缩放", "Z 标准化", "只用于尺度敏感算法")])
        st.subheader("入院诊断的多标签 0/1 编码", help="一个患者可以同时有多个入院诊断，每个诊断展开为独立的0/1列。")
        st.dataframe(dataframe_from_records(preview["admission_encoding"]["rows"]), hide_index=True, use_container_width=True)

elif scene == 7:
    title(7, "相关性分析与 LASSO", "先移除高度重复信息，再在训练折内部用L1逻辑回归选择稳定特征。")
    if require(eng.imputation_result is not None, "请先完成第5章缺失值实验。"):
        if st.button("运行嵌套五折 LASSO", type="primary"):
            progress_runner("特征筛选", eng.run_feature_selection)
        if eng.selection_result:
            result = eng.selection_result
            st.subheader("1. 高相关变量处理", help="每个外层训练折内计算Pearson相关系数；|r|≥0.85时移除重复信息，验证折不参与筛选。")
            st.dataframe(pd.DataFrame(result["correlation_filter"]), hide_index=True, use_container_width=True)
            st.subheader("2. λ 与交叉验证误差", help="λ越大，惩罚越强，更多系数被压到0。λmin误差最低；λ1se在最低误差一个标准误内选择更大的λ，通常更精简。")
            curve = pd.DataFrame(result["cv_curve"])
            st.line_chart(curve.set_index("log_lambda")[["mean_deviance"]], color="#0b8f83")
            metric_cards([("λmin", result["lambda_min"], "五折平均二项偏差最低"), ("λ1se", result["lambda_1se"], "最低误差一个标准误以内的最大λ"), ("最终规则", result["recommended_lambda"], None), ("入选特征数", result["selected_count"], None)])
            st.subheader("3. 最终非零系数", help="正负号表示模型中的关联方向，不代表疾病因果效应。")
            coef = pd.DataFrame(result["selected_coefficients"])
            st.bar_chart(coef.set_index("feature")[["coefficient"]], horizontal=True)
            st.dataframe(pd.DataFrame(result["folds"]), hide_index=True, use_container_width=True)

elif scene == 8:
    title(8, "模型比较", "所有方案使用同一套患者级五折：每轮4组训练、1组验证，同一患者绝不跨组。")
    if require(eng.selection_result is not None, "请先完成第7章的相关性分析与 LASSO。"):
        if st.button("训练并比较八种方案", type="primary"):
            progress_runner("模型比较与 SHAP", eng.run_compare)
        if eng.compare_result:
            result = eng.compare_result
            table = pd.DataFrame([{"模型": x["label"], "特征集": x["feature_set"], "验证AUC": x["metrics"]["roc_auc"], "训练AUC": x["training_auc"], "泛化差距": x["generalization_gap"], "折间波动": x["fold_sd"]} for x in result["models"]])
            st.dataframe(table.sort_values("验证AUC", ascending=False), hide_index=True, use_container_width=True)
            st.bar_chart(table.assign(方案=table["模型"]+" · "+table["特征集"]).set_index("方案")[["训练AUC","验证AUC"]])
            winner = result["winner"]
            st.success(f"课堂推荐：{winner['model']}（{winner['feature_set']}），折外 ROC AUC={winner['auc']:.3f}。{winner['rule']}")
            st.subheader("LASSO前后对各模型的影响", help="特征筛选不保证每个模型AUC都提高；还要同时看泛化差距、波动和特征数量。")
            st.dataframe(pd.DataFrame(result["pairwise_effects"]), hide_index=True, use_container_width=True)

elif scene == 9:
    title(9, "最优模型为什么这样判断", "SHAP把一次预测拆成各特征的推动作用；它解释模型，不证明因果。")
    if require(eng.compare_result is not None, "请先完成第8章模型比较，系统会解释实际胜出的模型。"):
        result = eng.global_explanation()
        st.subheader("全局 SHAP 重要性", help="平均绝对SHAP值越大，说明该变量通常对模型输出影响越大。")
        global_df = pd.DataFrame(result["global"]).head(15)
        st.bar_chart(global_df.set_index("feature")[["mean_abs"]], horizontal=True, color="#0b8f83")
        st.subheader("SHAP 蜂群图（简化散点）", help="横轴为SHAP值：大于0推动风险上升，小于0推动风险下降；每个点是一条患者记录。")
        bees = pd.DataFrame(result["beeswarm"])
        st.scatter_chart(bees, x="shap", y="feature", color="value", height=520)
        patient = st.selectbox("选择一个患者查看局部解释", result["patient_options"], format_func=lambda x: f"{x['patient_id']} · 概率 {x['probability']:.1%}")
        detail = eng.patient_explanation(patient["patient_id"])
        metric_cards([("基础风险", f"{detail['base_value']:.1%}", "所有患者的基准输出"), ("最终概率", f"{detail['probability']:.1%}", "基础风险加上各特征贡献后的输出")])
        waterfall = pd.DataFrame(detail["contributions"][:12])
        st.bar_chart(waterfall.set_index("feature")[["shap"]], horizontal=True)
        st.caption(result["warning"])

elif scene == 10:
    title(10, "PCA 与无监督探索", "隐藏出院标签后观察数据结构；聚类完成后再揭示金标准作事后对照。")
    if require(eng.compare_result is not None, "请先完成第8章模型比较。"):
        if st.button("运行 PCA 与 k=2–5 聚类", type="primary"):
            with st.spinner("正在标准化、降维并比较聚类数……"):
                st.session_state.unsupervised = eng.run_unsupervised()
        if "unsupervised" in st.session_state:
            result = st.session_state.unsupervised
            comp = pd.DataFrame(result["comparison"])
            metric_cards([("推荐簇数", result["recommended_k"], "来自当前数据的最高轮廓系数"), ("PC1解释率", f"{result['variance'][0]:.1%}", None), ("PC2解释率", f"{result['variance'][1]:.1%}", None)])
            st.subheader("k=2–5 的轮廓系数", help="轮廓系数越高，簇内越紧密、簇间越分离；它不是诊断准确率。")
            st.line_chart(comp.set_index("k")[["silhouette"]], color="#0b8f83")
            points = pd.DataFrame(result["points"])
            st.subheader("PCA 二维投影", help="PCA把多维入院特征投影到两个主成分，便于观察，但二维图会损失信息。")
            st.scatter_chart(points, x="x", y="y", color="cluster", height=560)
            st.subheader("聚类后揭示出院金标准")
            st.dataframe(pd.DataFrame(result["after_reveal"]), hide_index=True, use_container_width=True)
            st.info(result["message"])
