from __future__ import annotations

import hashlib
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.metrics import (accuracy_score, adjusted_rand_score, average_precision_score, balanced_accuracy_score, log_loss,
                             confusion_matrix, precision_recall_curve, precision_score, recall_score, roc_auc_score,
                             roc_curve, silhouette_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .generator import SEED


FIELD_LABELS = {"record_id":"记录号","patient_id":"患者编号","name":"姓名","id_number":"身份证号","sex_raw":"性别（原始）","age":"年龄","test_date":"入院日期","admission_diagnoses":"入院诊断","hgb_value":"血红蛋白（HGB）值","hgb_unit":"血红蛋白（HGB）单位","wbc":"白细胞计数（WBC）","rbc":"红细胞计数（RBC）","mcv":"平均红细胞体积（MCV）","mch":"平均红细胞血红蛋白量（MCH）","mchc":"平均红细胞血红蛋白浓度（MCHC）","rdw":"红细胞分布宽度（RDW）","plt":"血小板计数（PLT）","neut":"中性粒细胞计数（NEUT）","lymph":"淋巴细胞计数（LYMPH）","mpv":"平均血小板体积（MPV）","pdw":"血小板分布宽度（PDW）","monocyte":"单核细胞计数（MONO）","eosinophil":"嗜酸性粒细胞计数（EOS）","basophil":"嗜碱性粒细胞计数（BASO）","ferritin_ng_ml":"血清铁蛋白（ng/mL）","discharge_diagnosis":"最终出院诊断","anemia_treatment":"是否接受贫血治疗","note":"备注"}
BASE_CONTINUOUS = ["age_clean","hgb_gl","wbc","rbc","mcv","mch","mchc","rdw","plt","neut","lymph","nlr","mpv","pdw","monocyte","eosinophil","basophil"]
LINEAR_CONTINUOUS = BASE_CONTINUOUS + ["log_ferritin"]
TREE_CONTINUOUS = BASE_CONTINUOUS + ["ferritin_ng_ml"]
ALL_INPUT_COLUMNS = list(dict.fromkeys(BASE_CONTINUOUS + ["ferritin_ng_ml","log_ferritin","sex_binary"] + ["mcv_morph","admission_diagnoses"]))
# 兼容旧测试与无监督章节；监督学习会按算法选择原始或log铁蛋白。
NUMERIC_NO_SEX = LINEAR_CONTINUOUS
NUMERIC = LINEAR_CONTINUOUS + ["sex_binary"]
CAT = ["mcv_morph"]
ISSUE_ORDER = ["unit","age","sex","duplicate","critical","privacy","missing_hgb"]
ISSUES = {
    "unit":{"title":"血红蛋白（HGB）单位混用","correct":"convert","cause":"不同系统或导出模板的单位不一致。","risk":"g/dL 与 g/L 相差10倍。","options":[("convert","按 1 g/dL = 10 g/L 统一换算"),("divide","把 g/L 全部除以10"),("ignore","只保留数值，不看单位")]},
    "age":{"title":"年龄逻辑异常","correct":"missing","cause":"负年龄通常来自占位符、日期计算或录入错误。","risk":"会制造不存在的人群特征。","options":[("missing","设为缺失，进入后续填补实验"),("zero","改为0岁"),("median_now","在全数据上立刻填中位数")]},
    "sex":{"title":"性别编码不统一","correct":"map","cause":"中文、英文与数字编码来自不同字典。","risk":"同一性别被误认成多个类别。","options":[("map","男/M/1→1；女/F/0→0"),("onehot_raw","把男、M、1分别作为三个类别"),("drop","删除所有非中文编码记录")]},
    "duplicate":{"title":"已确认的重复导入","correct":"dedupe","cause":"同一条记录被重复导入。","risk":"会重复放大同一患者的权重。","options":[("dedupe","删除完全重复副本，保留首条"),("keep","全部保留"),("average","把重复检验值求平均")]},
    "critical":{"title":"血红蛋白（HGB）低于40 g/L","correct":"flag_keep","cause":"可能是真实危急值，也可能需要回查报告。","risk":"自动删除会丢失真正高风险患者。","options":[("flag_keep","核验后保留，并增加危急值标记"),("delete","为了让模型更稳定而删除"),("cap","直接截断到40 g/L")]},
    "privacy":{"title":"包含直接身份标识符","correct":"remove","cause":"业务数据含姓名和身份证号。","risk":"会带来隐私与记忆风险。","options":[("remove","从分析数据中移除姓名和身份证号"),("encode","把身份证号编码后进入模型"),("keep","保留姓名帮助模型记忆病例")]},
    "missing_hgb":{"title":"血红蛋白（HGB）存在缺失值","correct":"experiment","cause":"未检测、接口丢失或录入遗漏。","risk":"随意填0或全数据预填补会改变分布并泄漏信息。","options":[("experiment","保留缺失，下一章比较三种处理方法"),("zero","缺失值全部填0"),("drop_column","直接删除血红蛋白整列")]},
}


def records(frame: pd.DataFrame, cols: list[str] | None = None, limit: int | None = None) -> list[dict[str, Any]]:
    x = frame if cols is None else frame[cols]
    if limit is not None: x = x.head(limit)
    return x.replace([np.inf,-np.inf],np.nan).astype(object).where(pd.notna(x),None).to_dict("records")

def rnd(x: float, d: int=4) -> float: return round(float(x),d)
def feature_label(name: str) -> str:
    value=str(name).replace("continuous__","").replace("numeric__","").replace("sex__","").replace("mcv__mcv_morph_","MCV形态：").replace("admission__入院诊断_","入院诊断：")
    return {"age_clean":"年龄","sex_binary":"性别（0/1）","hgb_gl":"血红蛋白（HGB）","wbc":"白细胞计数（WBC）","rbc":"红细胞计数（RBC）","mcv":"平均红细胞体积（MCV）","mch":"平均红细胞血红蛋白量（MCH）","mchc":"平均红细胞血红蛋白浓度（MCHC）","rdw":"红细胞分布宽度（RDW）","plt":"血小板计数（PLT）","neut":"中性粒细胞计数（NEUT）","lymph":"淋巴细胞计数（LYMPH）","nlr":"中性粒细胞/淋巴细胞比值（NLR）","mpv":"平均血小板体积（MPV）","pdw":"血小板分布宽度（PDW）","monocyte":"单核细胞计数（MONO）","eosinophil":"嗜酸性粒细胞计数（EOS）","basophil":"嗜碱性粒细胞计数（BASO）","ferritin_ng_ml":"血清铁蛋白（原始值）","log_ferritin":"血清铁蛋白 log(1+x)"}.get(value,value)
def hist(s: pd.Series, bins: int=10) -> list[dict[str, Any]]:
    v=pd.to_numeric(s,errors="coerce").dropna().to_numpy(float)
    if not len(v): return []
    counts,edges=np.histogram(v,bins=bins); return [{"from":rnd(edges[i],2),"to":rnd(edges[i+1],2),"count":int(counts[i])} for i in range(len(counts))]
def metric(y: np.ndarray, p: np.ndarray, prob: np.ndarray) -> dict[str,float]:
    tn,fp,fn,tp=confusion_matrix(y,p,labels=[0,1]).ravel()
    return {"roc_auc":rnd(roc_auc_score(y,prob)),"accuracy":rnd(accuracy_score(y,p)),"balanced_accuracy":rnd(balanced_accuracy_score(y,p)),"sensitivity":rnd(recall_score(y,p,zero_division=0)),"specificity":rnd(tn/(tn+fp) if tn+fp else 0),"precision":rnd(precision_score(y,p,zero_division=0))}


class MultiDiagnosisEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        values = pd.Series(np.asarray(X).reshape(-1)).fillna("").astype(str)
        self.categories_ = sorted({item for value in values for item in value.split(";") if item})
        return self
    def transform(self, X):
        values=pd.Series(np.asarray(X).reshape(-1)).fillna("").astype(str)
        return np.asarray([[int(cat in set(value.split(";"))) for cat in self.categories_] for value in values],dtype=float)
    def get_feature_names_out(self, input_features=None):
        return np.asarray([f"入院诊断_{x}" for x in self.categories_],dtype=object)


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """在训练折内移除高度相关的连续变量，避免先看验证数据。"""
    def __init__(self, n_numeric: int, threshold: float=.85):
        self.n_numeric=n_numeric; self.threshold=threshold
    def fit(self, X, y=None):
        matrix=np.asarray(X,dtype=float); keep=np.ones(matrix.shape[1],dtype=bool); self.pairs_=[]
        corr=np.corrcoef(matrix[:,:self.n_numeric],rowvar=False)
        for i in range(self.n_numeric):
            for j in range(i+1,self.n_numeric):
                value=float(abs(corr[i,j])) if np.isfinite(corr[i,j]) else 0.0
                if keep[i] and keep[j] and value>=self.threshold:
                    keep[j]=False; self.pairs_.append((i,j,value))
        self.support_=keep; return self
    def transform(self, X): return np.asarray(X)[:,self.support_]
    def get_support(self, indices=False): return np.flatnonzero(self.support_) if indices else self.support_.copy()


@dataclass
class ClassroomEngine:
    data_path: Path
    lock: threading.RLock = field(default_factory=threading.RLock)
    def __post_init__(self): self.reset()
    def reset(self):
        with self.lock:
            self.raw=pd.read_csv(self.data_path); self.raw_hash=hashlib.sha256(self.data_path.read_bytes()).hexdigest(); self.issues={}; self.check_distributions={}; self.answers={}; self.model_frame=None; self.imputation_result=None; self.selection_result=None; self.compare_result=None; self.explanation_result=None; self.jobs={}; self.stage="raw"
            return self.state()
    def state(self):
        return {"stage":self.stage,"answered":len(self.answers),"answers":list(self.answers),"required":len(ISSUE_ORDER),"data_ready":self.model_frame is not None,"imputation_ready":self.imputation_result is not None,"selection_ready":self.selection_result is not None,"compare_ready":self.compare_result is not None}
    def raw_page(self,page:int=1,page_size:int=20):
        start=(max(page,1)-1)*page_size; d=self.raw["discharge_diagnosis"].value_counts()
        return {"state":self.state(),"summary":{"rows":len(self.raw),"patients":int(self.raw.patient_id.nunique()),"columns":len(self.raw.columns),"source":self.data_path.name,"raw_sha256":self.raw_hash,"discharge_distribution":[{"label":str(k),"count":int(v)} for k,v in d.items()]},"columns":[{"key":x,"label":FIELD_LABELS.get(x,x)} for x in self.raw.columns],"rows":records(self.raw.iloc[start:start+page_size]),"timeline":["入院血常规与入院诊断","风险提示模型","住院检查","最终出院诊断（金标准）"]}
    def run_checks(self):
        # 同一课堂中再次进入本页时返回既有扫描结果，避免导航返回造成“无法继续”。
        if self.stage != "raw" and self.issues:
            return {"state":self.state(),"issues":[self.issues[x] for x in ISSUE_ORDER],"distributions":self.check_distributions}
        if self.stage!="raw": raise ValueError("当前课堂已进入后续步骤；请从顶部导航返回已解锁章节。")
        f=self.raw; hgb=f.hgb_value.where(f.hgb_unit.eq("g/L"),f.hgb_value*10)
        masks={"unit":f.hgb_unit.ne("g/L"),"age":f.age.lt(0)|f.age.gt(120),"sex":~f.sex_raw.isin(["男","女"]),"duplicate":f.duplicated(keep=False),"critical":hgb.lt(40)&hgb.notna(),"privacy":f.name.notna()|f.id_number.notna(),"missing_hgb":f.hgb_value.isna()}
        self.issues={}
        for key in ISSUE_ORDER:
            cols=["record_id","patient_id","sex_raw","age","hgb_value","hgb_unit","admission_diagnoses","discharge_diagnosis"] if key!="privacy" else ["record_id","patient_id","name","id_number"]
            self.issues[key]={"id":key,**ISSUES[key],"affected_count":int(masks[key].sum()),"sample_records":records(f.loc[masks[key]],cols,6)}
        self.stage="checked"
        self.check_distributions={"units":[{"label":str(k),"count":int(v)} for k,v in f.hgb_unit.value_counts().items()],"ages":hist(f.age),"sex":[{"label":str(k),"count":int(v)} for k,v in f.sex_raw.value_counts().items()],"missing_positions":f.index[f.hgb_value.isna()].tolist()}
        return {"state":self.state(),"issues":[self.issues[x] for x in ISSUE_ORDER],"distributions":self.check_distributions}
    def answer_decision(self, issue_id:str, action:str):
        if self.stage not in {"checked","answering"} or issue_id not in self.issues: raise ValueError("请先完成数据检查。")
        issue=self.issues[issue_id]
        if action != issue["correct"]:
            wrong={"divide":"单位方向错误：会把本来正确的g/L缩小10倍。","ignore":"数值脱离单位没有临床含义。","zero":"0岁有真实含义，不能代表未知。","median_now":"会把验证患者的信息提前用于训练。","onehot_raw":"这会把同一性别拆成多个虚假类别。","drop":"删除编码变体会不必要地损失样本。","keep":"本题已确认是重复导入，保留会重复计算同一患者。","average":"完全重复记录不是重复检测结果，不能求平均。","delete":"危急值可能真实，删除会损失高风险患者。","cap":"截断会伪造检验结果。","encode":"编码身份证仍是可识别信息。","drop_column":"HGB是核心检验指标，不能因少量缺失删掉整列。"}
            return {"is_correct":False,"feedback":wrong.get(action,"该选项不符合本题的处理原则。"),"hint":f"提示：{issue['risk']} 请重新选择。","progress":len(self.answers)}
        self.answers[issue_id]=action; self.stage="answering"; next_id=next((x for x in ISSUE_ORDER if x not in self.answers),None)
        built=None
        if next_id is None: built=self.build_modeling_dataset()
        return {"is_correct":True,"feedback":"回答正确。该处理会进入分析审计。","hint":"","next_issue_id":next_id,"progress":len(self.answers),"built":built}
    def build_modeling_dataset(self):
        if len(self.answers)!=len(ISSUE_ORDER): raise ValueError("仍有题目未完成。")
        f=self.raw.copy(); f["hgb_gl"]=f.hgb_value.where(f.hgb_unit.eq("g/L"),f.hgb_value*10); f.loc[f.age.lt(0)|f.age.gt(120),"age"]=np.nan; f["age_clean"]=f.age; f["sex_binary"]=f.sex_raw.map({"男":1,"M":1,"1":1,"女":0,"F":0,"0":0}).astype(float); f=f.drop_duplicates(keep="first").copy(); f["critical_hgb_flag"]=(f.hgb_gl<40).astype(int); f=f.drop(columns=["name","id_number"]); f["nlr"]=f.neut/f.lymph.replace(0,np.nan); f["log_ferritin"]=np.log1p(pd.to_numeric(f.ferritin_ng_ml,errors="coerce")); f["mcv_morph"]=pd.cut(f.mcv,[-np.inf,80,100,np.inf],labels=["小细胞","正细胞","大细胞"],right=False).astype(str); f["thalassemia_label"]=(f.discharge_diagnosis=="地中海贫血").astype(int); f=f.dropna(subset=["patient_id","nlr"])
        self.model_frame=f; self.stage="data_ready"
        return {"rows_before":len(self.raw),"rows_after":len(f),"duplicates_removed":len(self.raw)-len(f),"raw_unchanged":hashlib.sha256(self.data_path.read_bytes()).hexdigest()==self.raw_hash,"audit":["HGB统一为g/L","非法年龄标为缺失","性别编码为0/1","删除完全重复副本","危急值保留并标记","移除身份标识","HGB缺失保留给填补实验","生成铁蛋白log(1+x)教学特征"],"preview":records(f,["record_id","age_clean","sex_binary","admission_diagnoses","hgb_gl","mcv","mch","rdw","ferritin_ng_ml","log_ferritin","discharge_diagnosis","thalassemia_label"],12)}
    def _frame(self):
        if self.model_frame is None: raise ValueError("请先完成数据处理。")
        return self.model_frame.copy()
    def _imputer(self, method:str):
        if method=="median": return SimpleImputer(strategy="median")
        if method=="knn": return KNNImputer(n_neighbors=5,weights="distance")
        if method=="iterative": return IterativeImputer(estimator=BayesianRidge(),random_state=SEED,max_iter=12)
        raise ValueError("未知填补方法")
    def _preprocessor(self, method:str, scale:bool, ferritin_mode:str="log"):
        steps=[("imputer",self._imputer(method))]
        if scale: steps.append(("scaler",StandardScaler()))
        continuous=Pipeline(steps)
        numeric = TREE_CONTINUOUS if ferritin_mode=="raw" else LINEAR_CONTINUOUS
        return ColumnTransformer([("continuous",continuous,numeric),("sex",SimpleImputer(strategy="most_frequent"),["sex_binary"]),("mcv",OneHotEncoder(handle_unknown="ignore"),CAT),("admission",MultiDiagnosisEncoder(),["admission_diagnoses"] )],sparse_threshold=0)
    def _split(self, frame):
        x=frame[ALL_INPUT_COLUMNS].copy(); y=frame.thalassemia_label.astype(int); groups=frame.patient_id.astype(str)
        return x,y,groups,list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED).split(x,y,groups))
    def _evaluate(self, frame:pd.DataFrame, model_key:str, method:str, selector_c:float|None=None, progress=None, progress_base=0, progress_span=100, selector_fold_cs:list[float]|None=None):
        specs={"logistic":("逻辑回归",LogisticRegression(max_iter=2500,random_state=SEED),True),"tree":("未剪枝决策树",DecisionTreeClassifier(random_state=SEED),False),"forest":("随机森林",RandomForestClassifier(n_estimators=260,min_samples_leaf=3,max_features="sqrt",random_state=SEED,n_jobs=-1),False)}
        label,estimator,scale=specs[model_key]; work=frame.copy()
        if method=="complete": work=work.dropna(subset=LINEAR_CONTINUOUS+["sex_binary"]).copy()
        x,y,groups,splits=self._split(work); prob=np.zeros(len(work)); train_auc=[]; folds=[]; selected_counts={}; correlation_removed={}
        for fold,(tr,te) in enumerate(splits,1):
            ferritin_mode="log" if (model_key=="logistic" or selector_c is not None) else "raw"
            fold_c=selector_fold_cs[fold-1] if selector_c is not None and selector_fold_cs else selector_c
            prep=self._preprocessor("median" if method=="complete" else method,scale or selector_c is not None,ferritin_mode); steps=[("prepare",prep)]
            if selector_c is not None:
                steps.append(("correlation",CorrelationFilter(len(LINEAR_CONTINUOUS),.85)))
                steps.append(("select",SelectFromModel(LogisticRegression(penalty="l1",solver="liblinear",C=fold_c,max_iter=2500,random_state=SEED),threshold=1e-8)))
            steps.append(("model",estimator)); pipe=Pipeline(steps); pipe.fit(x.iloc[tr],y.iloc[tr]); p=pipe.predict_proba(x.iloc[te])[:,1]; prob[te]=p; train_auc.append(roc_auc_score(y.iloc[tr],pipe.predict_proba(x.iloc[tr])[:,1])); pred=(p>=.5).astype(int)
            if selector_c is not None:
                names=pipe.named_steps["prepare"].get_feature_names_out(); corr_step=pipe.named_steps["correlation"]; kept=corr_step.get_support(indices=True); corr_names=names[kept]; chosen=pipe.named_steps["select"].get_support(indices=True)
                for left,right,value in corr_step.pairs_:
                    key=(names[left],names[right]); correlation_removed[key]=correlation_removed.get(key,0)+1
                for idx in chosen: selected_counts[corr_names[idx]]=selected_counts.get(corr_names[idx],0)+1
            folds.append({"fold":fold,"train_patients":int(groups.iloc[tr].nunique()),"validation_patients":int(groups.iloc[te].nunique()),"train_samples":int(len(tr)),"validation_samples":int(len(te)),"validation_auc":rnd(roc_auc_score(y.iloc[te],p)),"train_auc":rnd(train_auc[-1]),"patient_overlap":len(set(groups.iloc[tr])&set(groups.iloc[te]))})
            if progress: progress(progress_base+int(fold/5*progress_span),f"{label}：第 {fold}/5 折")
        pred=(prob>=.5).astype(int); m=metric(y.to_numpy(),pred,prob); fpr,tpr,_=roc_curve(y,prob); pr,re,_=precision_recall_curve(y,prob); tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
        final_mode="log" if (model_key=="logistic" or selector_c is not None) else "raw"
        final_steps=[("prepare",self._preprocessor("median" if method=="complete" else method,scale or selector_c is not None,final_mode))]
        if selector_c is not None:
            final_steps.append(("correlation",CorrelationFilter(len(LINEAR_CONTINUOUS),.85)))
            final_steps.append(("select",SelectFromModel(LogisticRegression(penalty="l1",solver="liblinear",C=selector_c,max_iter=2500,random_state=SEED),threshold=1e-8)))
        final_steps.append(("model",estimator)); final=Pipeline(final_steps).fit(x,y)
        selected_frequency=[{"feature":feature_label(k),"folds":v} for k,v in sorted(selected_counts.items(),key=lambda z:z[1],reverse=True)]
        correlation_summary=[{"kept":feature_label(a),"removed":feature_label(b),"folds":count,"reason":"训练折内 |r|≥0.85；保留原始指标，移除冗余派生指标"} for (a,b),count in sorted(correlation_removed.items(),key=lambda z:z[1],reverse=True)]
        return {"key":model_key,"label":label,"sample_count":len(work),"method":method,"selector_c":selector_c,"mean_selected_features":rnd(sum(selected_counts.values())/5,1) if selected_counts else None,"metrics":m,"training_auc":rnd(np.mean(train_auc)),"generalization_gap":rnd(max(0,np.mean(train_auc)-m["roc_auc"])),"folds":folds,"fold_sd":rnd(np.std([r["validation_auc"] for r in folds])),"confusion":{"tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp)},"roc":{"x":[rnd(x,4) for x in fpr],"y":[rnd(x,4) for x in tpr]},"pr":{"x":[rnd(x,4) for x in re],"y":[rnd(x,4) for x in pr]},"probabilities":[rnd(x,6) for x in prob],"truth":y.tolist(),"selected_frequency":selected_frequency,"correlation_filter":correlation_summary,"pipeline":final,"input":x,"frame":work}

    def _evaluate_lasso(self, frame:pd.DataFrame, method:str, selector_c:float, fold_cs:list[float], progress=None, progress_base=0, progress_span=100):
        """把LASSO作为最终逻辑回归模型评价，而不是先筛选后再换成另一种模型。"""
        work=frame.copy()
        if method=="complete": work=work.dropna(subset=LINEAR_CONTINUOUS+["sex_binary"]).copy()
        x,y,groups,splits=self._split(work); prob=np.zeros(len(work)); train_auc=[]; folds=[]; selected_counts={}
        for fold,(tr,te) in enumerate(splits,1):
            prep=self._preprocessor("median" if method=="complete" else method,True,"log")
            corr=CorrelationFilter(len(LINEAR_CONTINUOUS),.85)
            model=LogisticRegression(penalty="l1",solver="liblinear",C=float(fold_cs[fold-1]),max_iter=3000,random_state=SEED)
            pipe=Pipeline([("prepare",prep),("correlation",corr),("model",model)]).fit(x.iloc[tr],y.iloc[tr])
            p=pipe.predict_proba(x.iloc[te])[:,1]; prob[te]=p
            train_p=pipe.predict_proba(x.iloc[tr])[:,1]; train_auc.append(roc_auc_score(y.iloc[tr],train_p))
            names=pipe.named_steps["prepare"].get_feature_names_out()[pipe.named_steps["correlation"].get_support(indices=True)]
            support=np.abs(pipe.named_steps["model"].coef_[0])>1e-8
            for name in names[support]: selected_counts[name]=selected_counts.get(name,0)+1
            folds.append({"fold":fold,"train_patients":int(groups.iloc[tr].nunique()),"validation_patients":int(groups.iloc[te].nunique()),"train_samples":int(len(tr)),"validation_samples":int(len(te)),"validation_auc":rnd(roc_auc_score(y.iloc[te],p)),"train_auc":rnd(train_auc[-1]),"patient_overlap":len(set(groups.iloc[tr])&set(groups.iloc[te]))})
            if progress: progress(progress_base+int(fold/5*progress_span),f"LASSO逻辑回归：第 {fold}/5 折")
        pred=(prob>=.5).astype(int); metrics=metric(y.to_numpy(),pred,prob); fpr,tpr,_=roc_curve(y,prob); precision,recall,_=precision_recall_curve(y,prob); tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
        final=Pipeline([("prepare",self._preprocessor("median" if method=="complete" else method,True,"log")),("correlation",CorrelationFilter(len(LINEAR_CONTINUOUS),.85)),("model",LogisticRegression(penalty="l1",solver="liblinear",C=float(selector_c),max_iter=3000,random_state=SEED))]).fit(x,y)
        return {"key":"lasso","label":"LASSO逻辑回归","sample_count":len(work),"method":method,"selector_c":selector_c,"mean_selected_features":rnd(sum(selected_counts.values())/5,1),"metrics":metrics,"training_auc":rnd(np.mean(train_auc)),"generalization_gap":rnd(max(0,np.mean(train_auc)-metrics["roc_auc"])),"folds":folds,"fold_sd":rnd(np.std([r["validation_auc"] for r in folds])),"confusion":{"tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp)},"roc":{"x":[rnd(v,4) for v in fpr],"y":[rnd(v,4) for v in tpr]},"pr":{"x":[rnd(v,4) for v in recall],"y":[rnd(v,4) for v in precision]},"probabilities":[rnd(v,6) for v in prob],"truth":y.tolist(),"selected_frequency":[{"feature":feature_label(k),"folds":v} for k,v in sorted(selected_counts.items(),key=lambda z:z[1],reverse=True)],"correlation_filter":[],"pipeline":final,"input":x,"frame":work,"feature_set":"LASSO筛选后"}

    def _evaluate_svm(self, frame:pd.DataFrame, method:str, selector_c:float|None=None, selector_fold_cs:list[float]|None=None, progress=None, progress_base=0, progress_span=100):
        """外层五折评价；可在各外层训练折内先做LASSO筛选，再在内层四折选择SVM参数。"""
        work=frame.copy(); x,y,groups,splits=self._split(work); prob=np.zeros(len(work)); train_auc=[]; folds=[]; chosen=[]; selected_counts={}
        candidates=[(.1,"scale"),(.3,"scale"),(1,"scale"),(3,"scale"),(.3,.01),(1,.01),(3,.01),(.3,.03),(1,.03),(3,.03)]
        for fold,(tr,te) in enumerate(splits,1):
            fold_c=selector_fold_cs[fold-1] if selector_c is not None and selector_fold_cs else selector_c
            inner_x=x.iloc[tr].reset_index(drop=True); inner_y=y.iloc[tr].reset_index(drop=True); inner_g=groups.iloc[tr].reset_index(drop=True)
            inner_splits=list(StratifiedGroupKFold(n_splits=4,shuffle=True,random_state=SEED+fold).split(inner_x,inner_y,inner_g)); scored=[]
            for c,gamma in candidates:
                aucs=[]
                for a,b in inner_splits:
                    steps=[("prepare",self._preprocessor(method,True,"log"))]
                    if selector_c is not None:
                        steps.extend([("correlation",CorrelationFilter(len(LINEAR_CONTINUOUS),.85)),("select",SelectFromModel(LogisticRegression(penalty="l1",solver="liblinear",C=fold_c,max_iter=2500,random_state=SEED),threshold=1e-8))])
                    steps.append(("model",SVC(kernel="rbf",C=c,gamma=gamma,class_weight="balanced")))
                    pipe=Pipeline(steps).fit(inner_x.iloc[a],inner_y.iloc[a])
                    aucs.append(roc_auc_score(inner_y.iloc[b],pipe.decision_function(inner_x.iloc[b])))
                scored.append((float(np.mean(aucs)),c,gamma))
            best=max(scored,key=lambda z:(z[0],-z[1],0 if z[2]=="scale" else -float(z[2]))); _,c,gamma=best; chosen.append({"C":c,"gamma":gamma,"inner_auc":rnd(best[0])})
            steps=[("prepare",self._preprocessor(method,True,"log"))]
            if selector_c is not None:
                steps.extend([("correlation",CorrelationFilter(len(LINEAR_CONTINUOUS),.85)),("select",SelectFromModel(LogisticRegression(penalty="l1",solver="liblinear",C=fold_c,max_iter=2500,random_state=SEED),threshold=1e-8))])
            steps.append(("model",SVC(kernel="rbf",C=c,gamma=gamma,probability=True,class_weight="balanced",random_state=SEED)))
            pipe=Pipeline(steps).fit(x.iloc[tr],y.iloc[tr])
            p=pipe.predict_proba(x.iloc[te])[:,1]; prob[te]=p; train_p=pipe.predict_proba(x.iloc[tr])[:,1]; train_auc.append(roc_auc_score(y.iloc[tr],train_p))
            if selector_c is not None:
                names=pipe.named_steps["prepare"].get_feature_names_out()[pipe.named_steps["correlation"].get_support(indices=True)]
                for idx in pipe.named_steps["select"].get_support(indices=True): selected_counts[names[idx]]=selected_counts.get(names[idx],0)+1
            folds.append({"fold":fold,"train_patients":int(groups.iloc[tr].nunique()),"validation_patients":int(groups.iloc[te].nunique()),"train_samples":int(len(tr)),"validation_samples":int(len(te)),"validation_auc":rnd(roc_auc_score(y.iloc[te],p)),"train_auc":rnd(train_auc[-1]),"patient_overlap":len(set(groups.iloc[tr])&set(groups.iloc[te])),"selected_C":c,"selected_gamma":gamma})
            if progress: progress(progress_base+int(fold/5*progress_span),f"RBF-SVM{'（LASSO筛选后）' if selector_c is not None else ''}：外层第 {fold}/5 折")
        pred=(prob>=.5).astype(int); metrics=metric(y.to_numpy(),pred,prob); fpr,tpr,_=roc_curve(y,prob); precision,recall,_=precision_recall_curve(y,prob); tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
        # 展示模型使用外层折中最常出现的参数；最终成绩仍全部来自折外预测。
        parameter_pairs={(r["C"],str(r["gamma"])) for r in chosen}
        param=min(parameter_pairs,key=lambda p:(-sum(r["C"]==p[0] and str(r["gamma"])==p[1] for r in chosen),p[0],p[1])); final_gamma="scale" if param[1]=="scale" else float(param[1])
        final_steps=[("prepare",self._preprocessor(method,True,"log"))]
        if selector_c is not None:
            final_steps.extend([("correlation",CorrelationFilter(len(LINEAR_CONTINUOUS),.85)),("select",SelectFromModel(LogisticRegression(penalty="l1",solver="liblinear",C=selector_c,max_iter=2500,random_state=SEED),threshold=1e-8))])
        final_steps.append(("model",SVC(kernel="rbf",C=param[0],gamma=final_gamma,probability=True,class_weight="balanced",random_state=SEED)))
        final=Pipeline(final_steps).fit(x,y)
        return {"key":"svm","label":"支持向量机（RBF）","sample_count":len(work),"method":method,"selector_c":selector_c,"mean_selected_features":rnd(sum(selected_counts.values())/5,1) if selected_counts else None,"metrics":metrics,"training_auc":rnd(np.mean(train_auc)),"generalization_gap":rnd(max(0,np.mean(train_auc)-metrics["roc_auc"])),"folds":folds,"fold_sd":rnd(np.std([r["validation_auc"] for r in folds])),"confusion":{"tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp)},"roc":{"x":[rnd(v,4) for v in fpr],"y":[rnd(v,4) for v in tpr]},"pr":{"x":[rnd(v,4) for v in recall],"y":[rnd(v,4) for v in precision]},"probabilities":[rnd(v,6) for v in prob],"truth":y.tolist(),"selected_frequency":[{"feature":feature_label(k),"folds":v} for k,v in sorted(selected_counts.items(),key=lambda z:z[1],reverse=True)],"correlation_filter":[],"pipeline":final,"input":x,"frame":work,"feature_set":"LASSO筛选后" if selector_c is not None else "全部特征","tuning":{"method":"每个外层训练折内4折调参","grid":[{"C":c,"gamma":g} for c,g in candidates],"fold_choices":chosen}}
    def _ferritin_model_comparison(self, frame:pd.DataFrame) -> list[dict[str,Any]]:
        x,y,groups,splits=self._split(frame); method=self.imputation_result["recommended"] if self.imputation_result else "median"; rows=[]
        for mode,label in [("raw","使用原始铁蛋白"),("log","使用 log(1+铁蛋白)")]:
            oof=np.zeros(len(frame)); coefficients=[]
            for tr,te in splits:
                pipe=Pipeline([("prepare",self._preprocessor(method,True,mode)),("model",LogisticRegression(max_iter=2500,random_state=SEED))])
                pipe.fit(x.iloc[tr],y.iloc[tr]); oof[te]=pipe.predict_proba(x.iloc[te])[:,1]
                names=list(pipe.named_steps["prepare"].get_feature_names_out()); target="continuous__ferritin_ng_ml" if mode=="raw" else "continuous__log_ferritin"
                coefficients.append(float(pipe.named_steps["model"].coef_[0,names.index(target)]))
            rows.append({"mode":mode,"label":label,"roc_auc":rnd(roc_auc_score(y,oof)),"log_loss":rnd(log_loss(y,oof)),"coefficient_sd":rnd(np.std(coefficients)),"fold_coefficients":[rnd(v) for v in coefficients]})
        return rows
    def feature_preview(self):
        f=self._frame(); numeric=f[LINEAR_CONTINUOUS].copy(); med=numeric.median(); filled=numeric.fillna(med); scaler=StandardScaler().fit(filled); z=scaler.transform(filled); admission=MultiDiagnosisEncoder().fit(f[["admission_diagnoses"]]); encoded=admission.transform(f[["admission_diagnoses"]].head(8)); enc_cols=admission.get_feature_names_out().tolist(); enc_rows=pd.concat([f[["record_id","admission_diagnoses"]].head(8).reset_index(drop=True),pd.DataFrame(encoded[:8],columns=enc_cols)],axis=1)
        raw=f.ferritin_ng_ml.astype(float); logged=f.log_ferritin.astype(float)
        def summary(s):
            s=pd.Series(s).dropna().astype(float)
            return {"count":int(len(s)),"min":rnd(s.min(),2),"q1":rnd(s.quantile(.25),2),"median":rnd(s.median(),2),"q3":rnd(s.quantile(.75),2),"max":rnd(s.max(),2),"skewness":rnd(s.skew(),3)}
        catalog=[
            {"purpose":"处理缺失值","when":"部分记录缺少检验结果","common":"删除完整病例；中位数/众数；KNN填补","used":"中位数填补（训练折内）","active":True},
            {"purpose":"转换类别变量","when":"性别、入院诊断等不是连续数值","common":"0/1映射；独热编码","used":"性别0/1；入院诊断独热编码","active":True},
            {"purpose":"减弱右偏","when":"大量低值、少量极高值，分布有长右尾","common":"log(1+x)；平方根转换","used":"铁蛋白 log(1+x) 供逻辑回归/LASSO","active":True},
            {"purpose":"统一量纲","when":"线性模型、KNN、PCA依赖系数或距离","common":"Z标准化；Min–Max归一化","used":"逻辑回归/LASSO使用Z标准化","active":True},
            {"purpose":"处理异常值","when":"超出生理范围或可能是真实危急值","common":"回查；保留标记；有依据的截尾","used":"危急HGB人工核验后保留并标记","active":True},
            {"purpose":"构造派生变量","when":"临床关系不能由单个原始指标表达","common":"比值；分层；变化量","used":"NLR与MCV形态分类","active":True},
        ]
        return {"processing_catalog":catalog,"sex_mapping":[{"原始编码":"男 / M / 1","模型数值":1},{"原始编码":"女 / F / 0","模型数值":0}],"admission_encoding":{"rows":records(enc_rows),"columns":enc_rows.columns.tolist()},"mcv_encoding":records(f[["record_id","mcv","mcv_morph"]].head(8)),"missing":{"count":int(f.hgb_gl.isna().sum()),"positions":f.index[f.hgb_gl.isna()].tolist(),"before":hist(f.hgb_gl),"median_after":hist(f.hgb_gl.fillna(med.hgb_gl))},"ferritin_transform":{"formula":"log(1 + 血清铁蛋白)","raw":{"histogram":hist(raw),"summary":summary(raw)},"log":{"histogram":hist(logged),"summary":summary(logged)},"model_comparison":self._ferritin_model_comparison(f),"note":"log(1+x)压缩长右尾并减弱极端值对线性系数的影响；它不保证AUC一定提高。树模型仍使用原始铁蛋白。"},"standardization":{"method":"StandardScaler（Z 标准化）","z_scores":[{"feature":x,"raw":rnd(filled.iloc[0][x],3),"mean":rnd(scaler.mean_[i],3),"sd":rnd(scaler.scale_[i],3),"z":rnd(z[0,i],3)} for i,x in enumerate(LINEAR_CONTINUOUS)]},"correlation":{"features":LINEAR_CONTINUOUS,"matrix":[[rnd(v,3) for v in row] for row in filled.corr().to_numpy()]},"excluded":["discharge_diagnosis（最终金标准）","thalassemia_label（训练标签）","name / id_number（身份标识）","anemia_treatment（住院后信息）"]}
    def run_imputation(self,progress=None):
        f=self._frame(); result=[]
        for pos,(key,label) in enumerate([("complete","完整病例删除"),("median","中位数填补"),("knn","KNN填补")]):
            r=self._evaluate(f,"logistic",key,progress=progress,progress_base=pos*33,progress_span=33); result.append({k:v for k,v in r.items() if k not in {"pipeline","input","frame","probabilities","truth"}}|{"label":label})
        # 完整病例删除是对照：它会改变入组人群，不能仅因AUC较高就成为默认方案。
        fill_methods=[x for x in result if x["method"] != "complete"]
        best=max(x["metrics"]["roc_auc"] for x in fill_methods); median=next(x for x in result if x["method"]=="median"); candidates=[x for x in fill_methods if best-x["metrics"]["roc_auc"]<=.01]; recommended=median if best-median["metrics"]["roc_auc"]<=.01 else max(candidates,key=lambda x:x["metrics"]["roc_auc"])
        # 图形仅用于观察填补后的形状；模型成绩仍严格使用每一训练折拟合填补器。
        observed = f.hgb_gl.dropna()
        def summary(series):
            s=pd.Series(series).dropna().astype(float)
            return {"count":int(len(s)),"min":rnd(s.min(),1),"q1":rnd(s.quantile(.25),1),"median":rnd(s.median(),1),"q3":rnd(s.quantile(.75),1),"max":rnd(s.max(),1)}
        distributions=[]; filled_by_method={}
        for method,label in [("median","中位数填补"),("knn","KNN 填补")]:
            matrix=self._imputer(method).fit_transform(f[NUMERIC])
            hgb=pd.Series(matrix[:,NUMERIC.index("hgb_gl")])
            filled=hgb[f.hgb_gl.isna()]
            filled_by_method[method]=filled.to_numpy(float)
            distributions.append({"method":method,"label":label,"histogram":hist(hgb),"filled_summary":summary(filled),"completed_summary":summary(hgb)})
        median_auc=median["metrics"]["roc_auc"]; knn=next(x for x in result if x["method"]=="knn"); knn_auc=knn["metrics"]["roc_auc"]
        fold_rows=[]
        for fold in range(5):
            fold_rows.append({"fold":fold+1,"complete":result[0]["folds"][fold]["validation_auc"],"median":median["folds"][fold]["validation_auc"],"knn":knn["folds"][fold]["validation_auc"]})
        auc_explanation={"definition":"ROC AUC可理解为：随机抽取1名地中海贫血患者和1名非地中海贫血患者，模型把前者风险排得更高的概率。","calculation":"固定患者级5折；每轮用4折训练、1折预测。每名患者只产生1个折外预测概率，最后把全部折外预测拼接，再计算一次ROC AUC。它不是训练集AUC，也不是简单把5个折的AUC相加。","missing_records":int(f.hgb_gl.isna().sum()),"total_records":int(len(f)),"missing_percent":rnd(f.hgb_gl.isna().mean()*100,1),"median_auc":median_auc,"knn_auc":knn_auc,"auc_gap":rnd(abs(median_auc-knn_auc),4),"same_at_three_decimals":round(median_auc,3)==round(knn_auc,3),"filled_mean_absolute_difference":rnd(np.mean(np.abs(filled_by_method["median"]-filled_by_method["knn"])),2),"fold_rows":fold_rows,"why_close":"两种方法并非完全相同。HGB仅32/400条缺失，KNN估计值整体又接近中位数；同时模型还有MCV、MCH、RBC等信息。AUC只关心风险排序，所以少量概率变化未必改变很多患者对的先后顺序。"}
        self.imputation_result={"methods":result,"recommended":recommended["method"],"recommended_label":recommended["label"],"rule":"完整病例删除只作对照，因为它会改变入组人群；在中位数与KNN填补中，若中位数距最高AUC不超过0.01，优先选择更简单的中位数，否则选择AUC最高的方法。","distribution_note":"“填补前”统计的是原始已观测HGB；“新增填补值”只统计原来缺失的32项；“填补后全体”用于观察整体分布是否明显改变。AUC实验中的填补器始终只在训练折拟合。","observed_hgb_histogram":hist(observed),"observed_summary":summary(observed),"distributions":distributions,"auc_explanation":auc_explanation}
        return self.imputation_result
    def _prepare_lasso(self, x_train, x_test, method):
        prep=self._preprocessor(method,True,"log"); train=prep.fit_transform(x_train); test=prep.transform(x_test); names=np.asarray(prep.get_feature_names_out(),dtype=object)
        corr=CorrelationFilter(len(LINEAR_CONTINUOUS),.85).fit(train); kept=corr.get_support(indices=True)
        return corr.transform(train),corr.transform(test),names[kept],[(feature_label(names[i]),feature_label(names[j]),rnd(v,3)) for i,j,v in corr.pairs_],prep,corr
    def _choose_lambda(self, means, ses, lambdas, nonzero=None):
        means=np.asarray(means); ses=np.asarray(ses); lambdas=np.asarray(lambdas); valid=np.ones(len(lambdas),dtype=bool) if nonzero is None else np.asarray(nonzero)>0
        valid_idx=np.flatnonzero(valid); min_i=int(valid_idx[np.argmin(means[valid_idx])]); cutoff=means[min_i]+ses[min_i]; eligible=np.flatnonzero(valid & (means<=cutoff+1e-12)); one_i=int(eligible[np.argmax(lambdas[eligible])])
        return min_i,one_i
    def _lasso_cv_curve(self, x, y, groups, method, lambdas):
        split=list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED).split(x,y,groups)); scores=np.zeros((len(split),len(lambdas)))
        for fold,(tr,te) in enumerate(split):
            xt,xv,_,_,_,_=self._prepare_lasso(x.iloc[tr],x.iloc[te],method)
            for j,lam in enumerate(lambdas):
                model=LogisticRegression(penalty="l1",solver="liblinear",C=1/float(lam),max_iter=3000,random_state=SEED).fit(xt,y.iloc[tr]); scores[fold,j]=log_loss(y.iloc[te],model.predict_proba(xv)[:,1],labels=[0,1])
        return scores.mean(axis=0),scores.std(axis=0,ddof=1)/np.sqrt(len(split))
    def run_feature_selection(self,progress=None):
        if self.imputation_result is None: raise ValueError("请先完成缺失值实验。")
        f=self._frame(); method=self.imputation_result["recommended"]; x,y,groups,outer_splits=self._split(f); lambdas=np.logspace(-2,1.7,45)
        if progress: progress(3,"准备患者级嵌套交叉验证")
        candidates=[]; frequency_min={}; frequency_1se={}; outer_min_cs=[]; outer_1se_cs=[]; correlation_counts={}
        for fold,(tr,te) in enumerate(outer_splits,1):
            inner_x=x.iloc[tr].reset_index(drop=True); inner_y=y.iloc[tr].reset_index(drop=True); inner_g=groups.iloc[tr].reset_index(drop=True)
            means,ses=self._lasso_cv_curve(inner_x,inner_y,inner_g,method,lambdas); min_i,one_i=self._choose_lambda(means,ses,lambdas)
            lambda_min_fold=float(lambdas[min_i]); lambda_1se_fold=float(lambdas[one_i]); outer_min_cs.append(1/lambda_min_fold); outer_1se_cs.append(1/lambda_1se_fold)
            xt,xv,names,pairs,_,_=self._prepare_lasso(x.iloc[tr],x.iloc[te],method)
            model_min=LogisticRegression(penalty="l1",solver="liblinear",C=1/lambda_min_fold,max_iter=3000,random_state=SEED).fit(xt,y.iloc[tr])
            model_1se=LogisticRegression(penalty="l1",solver="liblinear",C=1/lambda_1se_fold,max_iter=3000,random_state=SEED).fit(xt,y.iloc[tr])
            prob_min=model_min.predict_proba(xv)[:,1]; prob_1se=model_1se.predict_proba(xv)[:,1]
            support_min=np.abs(model_min.coef_[0])>1e-8; support_1se=np.abs(model_1se.coef_[0])>1e-8
            for name in names[support_min]: frequency_min[feature_label(name)]=frequency_min.get(feature_label(name),0)+1
            for name in names[support_1se]: frequency_1se[feature_label(name)]=frequency_1se.get(feature_label(name),0)+1
            for kept,removed,value in pairs: correlation_counts[(kept,removed,value)]=correlation_counts.get((kept,removed,value),0)+1
            candidates.append({"fold":fold,"train_patients":int(groups.iloc[tr].nunique()),"validation_patients":int(groups.iloc[te].nunique()),"patient_overlap":len(set(groups.iloc[tr])&set(groups.iloc[te])),"lambda_min":rnd(lambda_min_fold,6),"lambda_1se":rnd(lambda_1se_fold,6),"selected_count_min":int(support_min.sum()),"selected_count_1se":int(support_1se.sum()),"auc_min":rnd(roc_auc_score(y.iloc[te],prob_min)),"auc_1se":rnd(roc_auc_score(y.iloc[te],prob_1se)),"selected_features_min":[feature_label(v) for v in names[support_min]],"selected_features_1se":[feature_label(v) for v in names[support_1se]]})
            if progress: progress(8+fold*11,f"外层第 {fold}/5 折：训练折内选择 λ")
        means,ses=self._lasso_cv_curve(x,y,groups,method,lambdas)
        full_train,_,names,pairs,_,_=self._prepare_lasso(x,x.iloc[:1],method); path=[]; nonzero=[]
        for lam in lambdas:
            model=LogisticRegression(penalty="l1",solver="liblinear",C=1/float(lam),max_iter=3000,random_state=SEED).fit(full_train,y); coef=model.coef_[0]; nonzero.append(int((np.abs(coef)>1e-8).sum())); path.append([rnd(v,6) for v in coef])
        min_i,one_i=self._choose_lambda(means,ses,lambdas,nonzero); lambda_min=float(lambdas[min_i]); lambda_1se=float(lambdas[one_i])
        coef_min=np.asarray(path[min_i]); coef_1se=np.asarray(path[one_i]); selected_min=[feature_label(names[i]) for i in np.flatnonzero(np.abs(coef_min)>1e-8)]; selected_1se=[feature_label(names[i]) for i in np.flatnonzero(np.abs(coef_1se)>1e-8)]
        correlations=[]
        for kept,removed,value in pairs:
            reason="NLR由中性粒细胞和淋巴细胞计算；达到阈值时优先保留临床含义更直接的原始指标。" if "NLR" in removed else "两者提供高度重复的线性信息，保留排序靠前且临床含义更直接的指标。"
            correlations.append({"kept":kept,"removed":removed,"correlation":value,"reason":reason})
        mean_min=float(np.mean([r["auc_min"] for r in candidates])); mean_1se=float(np.mean([r["auc_1se"] for r in candidates]))
        use_1se=mean_min-mean_1se<=.01; chosen="lambda_1se" if use_1se else "lambda_min"
        chosen_coef=coef_1se if use_1se else coef_min; chosen_selected=selected_1se if use_1se else selected_min
        chosen_lambda=lambda_1se if use_1se else lambda_min; chosen_cs=outer_1se_cs if use_1se else outer_min_cs; chosen_frequency=frequency_1se if use_1se else frequency_min
        outer_rows=[]
        for row in candidates:
            suffix="1se" if use_1se else "min"
            outer_rows.append({**row,"selected_count":row[f"selected_count_{suffix}"],"validation_auc":row[f"auc_{suffix}"],"selected_features":row[f"selected_features_{suffix}"]})
        if progress: progress(96,"整理系数路径与最终入选变量")
        selected_coefficients=[{"feature":feature_label(names[i]),"coefficient":rnd(chosen_coef[i],6),"direction":"提高预测概率" if chosen_coef[i]>0 else "降低预测概率"} for i in np.flatnonzero(np.abs(chosen_coef)>1e-8)]
        excluded=[feature_label(names[i]) for i in np.flatnonzero(np.abs(chosen_coef)<=1e-8)]
        reason="λ1se与λmin的外层AUC差不超过0.01，优先采用更精简的λ1se。" if use_1se else "λ1se的外层AUC比λmin低超过0.01，因此采用验证表现更好的λmin，避免过度收缩。"
        self.selection_result={"method":"相关性预筛 + 标准LASSO（L1逻辑回归）","lambda_sequence":[rnd(v,8) for v in lambdas],"coefficient_path":{"features":[feature_label(v) for v in names],"coefficients":path,"nonzero_counts":nonzero},"cv_curve":[{"lambda":rnd(lambdas[i],8),"log_lambda":rnd(np.log(lambdas[i]),5),"mean_deviance":rnd(means[i],6),"standard_error":rnd(ses[i],6),"nonzero_count":nonzero[i]} for i in range(len(lambdas))],"lambda_min":rnd(lambda_min,8),"lambda_1se":rnd(lambda_1se,8),"recommended_lambda":chosen,"recommended_c":1/chosen_lambda,"outer_fold_cs":chosen_cs,"outer_fold_min_cs":outer_min_cs,"outer_fold_1se_cs":outer_1se_cs,"selected_min":selected_min,"selected_1se":selected_1se,"selected_default":chosen_selected,"selected_count":len(chosen_selected),"selected_coefficients":selected_coefficients,"excluded_features":excluded,"folds":outer_rows,"correlation_filter":correlations,"frequency":[{"feature":k,"folds":v} for k,v in sorted(chosen_frequency.items(),key=lambda z:(-z[1],z[0]))],"nested_auc":rnd(np.mean([r["validation_auc"] for r in outer_rows])),"lambda_comparison":{"mean_outer_auc_min":rnd(mean_min),"mean_outer_auc_1se":rnd(mean_1se),"chosen":chosen},"message":reason+" LASSO系数表示模型关联方向，不是疾病因果效应。"}
        return self.selection_result
    def _tree_lab(self,f,method,selector_c):
        rows=[]
        for d in [2,3,4,5,6,8,10,None]:
            # Reuse a compact manual tree CV for the teaching curve.
            x,y,g,splits=self._split(f); aucs=[]; train=[]; nodes=[]
            for tr,te in splits:
                prep=self._preprocessor(method,True); xt=prep.fit_transform(x.iloc[tr]); xv=prep.transform(x.iloc[te]); sel=SelectFromModel(LogisticRegression(penalty="l1",solver="liblinear",C=selector_c,max_iter=2500,random_state=SEED),threshold=1e-8).fit(xt,y.iloc[tr]); tree=DecisionTreeClassifier(max_depth=d,min_samples_leaf=6,random_state=SEED).fit(sel.transform(xt),y.iloc[tr]); aucs.append(roc_auc_score(y.iloc[te],tree.predict_proba(sel.transform(xv))[:,1])); train.append(roc_auc_score(y.iloc[tr],tree.predict_proba(sel.transform(xt))[:,1])); nodes.append(tree.tree_.node_count)
            rows.append({"depth":"不限制" if d is None else d,"train_auc":rnd(np.mean(train)),"validation_auc":rnd(np.mean(aucs)),"nodes":rnd(np.mean(nodes),1)})
        return rows
    def run_compare(self,progress=None):
        if self.selection_result is None: raise ValueError("请先完成特征筛选。")
        f=self._frame(); method=self.imputation_result["recommended"]; selector_c=self.selection_result["recommended_c"]; fold_cs=self.selection_result.get("outer_fold_cs"); rows=[]
        # 每个模型都同时评价全部特征与“在当前外层训练折内完成LASSO筛选”的特征集。
        rows.append(self._evaluate(f,"logistic",method,progress=progress,progress_base=0,progress_span=12)|{"feature_set":"全部特征"})
        rows.append(self._evaluate_lasso(f,method,selector_c,fold_cs,progress=progress,progress_base=12,progress_span=12))
        rows.append(self._evaluate(f,"tree",method,progress=progress,progress_base=24,progress_span=12)|{"feature_set":"全部特征"})
        rows.append(self._evaluate(f,"tree",method,selector_c,progress=progress,progress_base=36,progress_span=12,selector_fold_cs=fold_cs)|{"feature_set":"LASSO筛选后"})
        rows.append(self._evaluate(f,"forest",method,progress=progress,progress_base=48,progress_span=12)|{"feature_set":"全部特征"})
        rows.append(self._evaluate(f,"forest",method,selector_c,progress=progress,progress_base=60,progress_span=12,selector_fold_cs=fold_cs)|{"feature_set":"LASSO筛选后"})
        rows.append(self._evaluate_svm(f,method,progress=progress,progress_base=72,progress_span=13))
        rows.append(self._evaluate_svm(f,method,selector_c,fold_cs,progress=progress,progress_base=85,progress_span=13))
        best_auc=max(r["metrics"]["roc_auc"] for r in rows); eligible=[r for r in rows if best_auc-r["metrics"]["roc_auc"]<=.01]
        winner=min(eligible,key=lambda r:(r["generalization_gap"],r["fold_sd"],0 if r["selector_c"] is not None else 1,-r["metrics"]["roc_auc"]))
        logistic_full=rows[0]; lasso=rows[1]
        selection_effect={"auc_change":rnd(lasso["metrics"]["roc_auc"]-logistic_full["metrics"]["roc_auc"]),"gap_change":rnd(lasso["generalization_gap"]-logistic_full["generalization_gap"]),"train_auc_change":rnd(lasso["training_auc"]-logistic_full["training_auc"]),"full_auc":logistic_full["metrics"]["roc_auc"],"lasso_auc":lasso["metrics"]["roc_auc"],"full_gap":logistic_full["generalization_gap"],"lasso_gap":lasso["generalization_gap"],"improved_auc":lasso["metrics"]["roc_auc"]>logistic_full["metrics"]["roc_auc"],"reduced_gap":lasso["generalization_gap"]<logistic_full["generalization_gap"]}
        pairwise_effects=[]
        for key,label in [("logistic","逻辑回归"),("tree","决策树"),("forest","随机森林"),("svm","支持向量机")]:
            full=next(r for r in rows if r["key"]==key and r["feature_set"]=="全部特征")
            selected=lasso if key=="logistic" else next(r for r in rows if r["key"]==key and r["feature_set"]=="LASSO筛选后")
            pairwise_effects.append({"key":key,"model":label,"full_auc":full["metrics"]["roc_auc"],"selected_auc":selected["metrics"]["roc_auc"],"auc_change":rnd(selected["metrics"]["roc_auc"]-full["metrics"]["roc_auc"]),"full_gap":full["generalization_gap"],"selected_gap":selected["generalization_gap"],"gap_change":rnd(selected["generalization_gap"]-full["generalization_gap"]),"selected_features":selected["mean_selected_features"]})
        self.compare_result={"models":[{k:v for k,v in r.items() if k not in {"pipeline","input","frame"}} for r in rows],"selection_effect":selection_effect,"pairwise_effects":pairwise_effects,"winner":{"key":winner["key"],"model":winner["label"],"feature_set":winner["feature_set"],"auc":winner["metrics"]["roc_auc"],"selector_c":winner["selector_c"],"rule":"先保留AUC距最高值不超过0.01的方案，再依次优先选择训练—验证差距更小、折间波动更小、特征更少的模型。"}}; self._build_explanation(winner); return self.compare_result
    def _build_explanation(self,winner):
        import shap
        pipe=winner["pipeline"]; x=winner["input"]; matrix=pipe.named_steps["prepare"].transform(x); names=list(pipe.named_steps["prepare"].get_feature_names_out())
        if "correlation" in pipe.named_steps:
            kept=pipe.named_steps["correlation"].get_support(indices=True); matrix=pipe.named_steps["correlation"].transform(matrix); names=[names[i] for i in kept]
        if "select" in pipe.named_steps:
            chosen=pipe.named_steps["select"].get_support(indices=True); matrix=pipe.named_steps["select"].transform(matrix); names=[names[i] for i in chosen]
        model=pipe.named_steps["model"]
        if isinstance(model,(RandomForestClassifier,DecisionTreeClassifier)):
            explainer=shap.TreeExplainer(model,data=matrix[:100],model_output="probability",feature_perturbation="interventional"); values=explainer.shap_values(matrix); values=np.asarray(values[1] if isinstance(values,list) else values); values=values[:,:,1] if values.ndim==3 else values; base=float(np.asarray(explainer.expected_value).reshape(-1)[-1])
        else:
            explainer=shap.LinearExplainer(model,matrix[:100]); values=np.asarray(explainer.shap_values(matrix)); base=float(np.asarray(explainer.expected_value).reshape(-1)[0])
        mean=np.abs(values).mean(axis=0); order=np.argsort(mean)[::-1]; probs=model.predict_proba(matrix)[:,1]; idxs=np.linspace(0,len(x)-1,min(140,len(x))).astype(int)
        clean=[feature_label(n) for n in names]
        self.explanation_result={"model":winner["label"],"base_value":rnd(base,6),"global":[{"feature":clean[i],"mean_abs":rnd(mean[i],6)} for i in order],"beeswarm":[{"feature":clean[j],"shap":rnd(values[i,j],6),"value":rnd(matrix[i,j],4)} for i in idxs for j in order[:10]],"patients":[{"patient_id":str(winner["frame"].patient_id.iloc[i]),"record_id":str(winner["frame"].record_id.iloc[i]),"probability":rnd(probs[i],6),"raw":records(winner["frame"].iloc[[i]], ["age_clean","sex_binary","admission_diagnoses","hgb_gl","rbc","mcv","mch","rdw"])[0],"contributions":sorted([{"feature":clean[j],"value":rnd(matrix[i,j],4),"shap":rnd(values[i,j],6)} for j in range(len(clean))],key=lambda z:abs(z["shap"]),reverse=True)} for i in range(len(x))],"warning":"SHAP解释模型给出该概率的原因，不代表疾病因果机制。"}
    def global_explanation(self):
        if not self.explanation_result: raise ValueError("请先完成模型比较。")
        return {k:v for k,v in self.explanation_result.items() if k!="patients"}|{"patient_options":[{"patient_id":x["patient_id"],"record_id":x["record_id"],"probability":x["probability"]} for x in self.explanation_result["patients"]]}
    def patient_explanation(self,patient_id:str):
        if not self.explanation_result: raise ValueError("请先完成模型比较。")
        row=next((x for x in self.explanation_result["patients"] if x["patient_id"]==patient_id),None)
        if not row: raise ValueError("未找到患者。")
        return row|{"base_value":self.explanation_result["base_value"],"warning":self.explanation_result["warning"]}
    def evaluate_threshold(self,threshold:float):
        if not self.compare_result: raise ValueError("请先完成模型比较。")
        # Find the stored winner by re-evaluating from the explanation source is unnecessary; preserve OOF result.
        meta=self.compare_result["winner"]; winner=next(r for r in self.compare_result["models"] if r["key"]==meta["key"] and r["feature_set"]==meta["feature_set"]); y=np.asarray(winner["truth"]); p=np.asarray(winner["probabilities"]); pred=(p>=threshold).astype(int); return {"threshold":rnd(threshold,2),"metrics":metric(y,pred,p),"confusion":dict(zip(["tn","fp","fn","tp"],[int(x) for x in confusion_matrix(y,pred,labels=[0,1]).ravel()]))}
    def run_unsupervised(self):
        if not self.compare_result: raise ValueError("请先完成模型比较。")
        f=self._frame(); unsupervised_features=BASE_CONTINUOUS+["ferritin_ng_ml","sex_binary"]; base=f[unsupervised_features].copy().fillna(f[unsupervised_features].median()); z=StandardScaler().fit_transform(base); pca=PCA(n_components=2,random_state=SEED).fit(z); coords=pca.transform(z); comparisons=[]; results={}
        for k in range(2,6):
            km=KMeans(n_clusters=k,n_init=20,random_state=SEED).fit(z); labels=km.labels_; comparisons.append({"k":k,"silhouette":rnd(silhouette_score(z,labels)),"inertia":rnd(km.inertia_,2),"ari_after_reveal":rnd(adjusted_rand_score(f.thalassemia_label,labels))}); results[k]=labels
        best=max(comparisons,key=lambda x:x["silhouette"]); labels=results[best["k"]]
        return {"recommended_k":best["k"],"comparison":comparisons,"points":[{"x":rnd(coords[i,0]),"y":rnd(coords[i,1]),"cluster":int(labels[i]),"gold_standard":int(f.thalassemia_label.iloc[i])} for i in range(len(f))],"variance":[rnd(x) for x in pca.explained_variance_ratio_],"loadings":[{"feature":x,"pc1":rnd(pca.components_[0,i]),"pc2":rnd(pca.components_[1,i])} for i,x in enumerate(unsupervised_features)],"after_reveal":[{"cluster":int(c),"size":int((labels==c).sum()),"thalassemia_rate":rnd(f.loc[labels==c,"thalassemia_label"].mean())} for c in range(best["k"])],"message":"PCA仅用于投影观察；选择k时同时查看轮廓系数与肘部曲线，簇不等于疾病名称。"}
    def start_job(self,kind:str):
        runs={"imputation":self.run_imputation,"selection":self.run_feature_selection,"compare":self.run_compare}
        if kind not in runs: raise ValueError("未知任务。")
        active=next((x for x in self.jobs.values() if x["kind"]==kind and x["status"] in {"queued","running"}),None)
        if active:return dict(active)
        job={"job_id":uuid.uuid4().hex,"kind":kind,"status":"queued","progress":0,"message":"等待开始","result":None,"error":None}; self.jobs[job["job_id"]]=job
        def worker():
            job.update(status="running",message="正在准备数据")
            try:
                result=runs[kind](lambda p,m:job.update(progress=int(p),message=m)); job.update(status="completed",progress=100,message="计算完成",result=result)
            except Exception as exc: job.update(status="failed",message="运行失败",error=str(exc))
        threading.Thread(target=worker,daemon=True).start(); return dict(job)
    def job_status(self,job_id):
        if job_id not in self.jobs: raise ValueError("找不到任务。")
        return dict(self.jobs[job_id])
