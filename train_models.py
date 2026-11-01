from pathlib import Path
import pandas as pd, json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
ROOT=Path(__file__).resolve().parent; OUT=ROOT/"outputs"; OUT.mkdir(exist_ok=True)
df=pd.read_csv(ROOT/"data/student_performance.csv"); df["pass_status"]=(df.final_score>=60).astype(int)
features=["study_hours_per_day","attendance_percent","previous_exam_score","assignments_completed","sleep_hours","internet_access","parental_support"]
numeric=features[:5]; categorical=features[5:]; X=df[features]
def prep():
    return ColumnTransformer([("num",Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())]),numeric),("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),categorical)])
Xtr,Xte,yrtr,yrte,yctr,ycte=train_test_split(
    X,df.final_score,df.pass_status,test_size=.2,random_state=42,stratify=df.pass_status
)
reg=[]
for name,model in {"Linear Regression":LinearRegression(),"Random Forest Regressor":RandomForestRegressor(n_estimators=200,random_state=42)}.items():
    p=Pipeline([("prep",prep()),("model",model)]); p.fit(Xtr,yrtr); pred=p.predict(Xte)
    reg.append({"model":name,"MAE":mean_absolute_error(yrte,pred),"RMSE":mean_squared_error(yrte,pred)**.5,"R2":r2_score(yrte,pred)})
rdf=pd.DataFrame(reg).sort_values("RMSE"); rdf.to_csv(OUT/"regression_model_comparison.csv",index=False)
cl=[]
for name,model in {"Logistic Regression":LogisticRegression(max_iter=1000),"Random Forest Classifier":RandomForestClassifier(n_estimators=200,random_state=42)}.items():
    p=Pipeline([("prep",prep()),("model",model)]); p.fit(Xtr,yctr); pred=p.predict(Xte)
    cl.append({"model":name,"Accuracy":accuracy_score(ycte,pred),"Precision":precision_score(ycte,pred,zero_division=0),"Recall":recall_score(ycte,pred,zero_division=0),"F1":f1_score(ycte,pred,zero_division=0)})
    if len(cl)==1 or cl[-1]["F1"]>max(x["F1"] for x in cl[:-1]):
        best_cm=confusion_matrix(ycte,pred).tolist(); best_report=classification_report(ycte,pred,target_names=["At Risk (Fail)","Pass"],zero_division=0); best_name=name
cdf=pd.DataFrame(cl).sort_values("F1",ascending=False); cdf.to_csv(OUT/"classification_model_comparison.csv",index=False)
(OUT/"best_classification_report.txt").write_text("Best model: "+best_name+"\n\n"+best_report,encoding="utf-8")
(OUT/"best_confusion_matrix.json").write_text(json.dumps({"labels":["At Risk (Fail)","Pass"],"matrix":best_cm},indent=2))
plt.figure(figsize=(7,5)); plt.scatter(df.study_hours_per_day,df.final_score,alpha=.5); plt.xlabel("Study hours/day"); plt.ylabel("Final score"); plt.title("Study Hours vs Final Score"); plt.tight_layout(); plt.savefig(OUT/"study_hours_vs_final_score.png"); plt.close()
for fn,data,metric,title in [("regression_rmse_comparison.png",rdf,"RMSE","Regression RMSE (lower is better)"),("classification_f1_comparison.png",cdf,"F1","Classification F1 (higher is better)")]:
    plt.figure(figsize=(7,4)); plt.bar(data.model,data[metric]); plt.ylabel(metric); plt.title(title); plt.xticks(rotation=12,ha="right"); plt.tight_layout(); plt.savefig(OUT/fn); plt.close()
(OUT/"run_summary.json").write_text(json.dumps({"rows":len(df),"data":"synthetic educational demonstration data","best_regression":rdf.iloc[0].to_dict(),"best_classification":cdf.iloc[0].to_dict(),"pass_threshold":40},indent=2,default=float))
print("Student Performance Prediction completed. Outputs saved in outputs/")
print(rdf.to_string(index=False)); print(cdf.to_string(index=False))
