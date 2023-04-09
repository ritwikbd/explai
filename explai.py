# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 19:05:48 2023

@author: ritwi
"""

from reportlab.pdfgen import canvas
from reportlab.lib.units import inch, cm
from reportlab.lib.utils import ImageReader

from sklearn import datasets
import matplotlib
import seaborn as sns

import pandas as pd
matplotlib.pyplot.ticklabel_format(style='plain', axis='y')


def getPcaPlot(df_x,yval,title,image_name):
    
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)

    pca_df=pd.DataFrame(pca.fit_transform(df_x),columns=['feat1','feat2'])
    pca_comps=pca.explained_variance_ratio_
    pca_text= "Feature 1 variance capture =" + str(round(pca_comps[0],3)) + \
    "\n"+ "Feature 2 variance capture =" + str(round(pca_comps[1],3))                                         
    
    
    sns.set(font='Verdana')
    t=sns.scatterplot(x=pca_df['feat1'],y=pca_df['feat2'],hue=list(yval))
    t.set_xlabel("feat1="+str(round(pca_comps[0],3)),fontsize=15)
    t.set_ylabel("feat2="+str(round(pca_comps[1],3)),fontsize=15)

    t.set_xticklabels(t.get_xticks())
    t.set_yticklabels(t.get_yticks())
    t.axes.set_title(title)
    img=t.get_figure()
    img.savefig(image_name,bbox_inches='tight')
    t.remove()
    matplotlib.pyplot.close()
    return pca
    
def getEvals_clf(df_y,preds,predsio):
    from sklearn import metrics
    evals=['','','','']
    evals[0] = str(round(metrics.accuracy_score(df_y,preds),4))
    evals[1] = str(round(metrics.log_loss(df_y,predsio),4))
    if len(df_y.value_counts()) == 2:
        evals[2] = str(round(metrics.precision_score(df_y,preds),4))
        evals[3] = str(round(metrics.recall_score(df_y,preds),4))
    else:
        evals[2] = 'NA'
        evals[3] = 'NA'
    return evals

def getConfusionMatrix(df_y,preds):
    from sklearn import metrics
    p=metrics.confusion_matrix(df_y,preds)
    t2=sns.heatmap(p,annot=True,fmt='g')
    t2.set_xlabel("Predicted")
    t2.set_ylabel("Actual")
    img2=t2.get_figure()
    img2.savefig("heatmapConfMatrix.png",bbox_inches='tight')
    t2.remove()
    matplotlib.pyplot.close()

def standardFeatImps(pca,df_x):
    feat_cols_imp=pca.components_
    pca_cols_imp=pca.explained_variance_ratio_
    feats_dict={}
    all_cols=list(df_x.columns)
    for co in all_cols:
        ind=list(df_x).index(co)
        imp=(pca_cols_imp[0]*abs(feat_cols_imp[0][ind]) + pca_cols_imp[1]*abs(feat_cols_imp[1][ind]))/sum(pca_cols_imp)
        feats_dict[co] =round(imp,4) * 100
    df_featimp2=pd.DataFrame(feats_dict.items(),columns=['feature','importance'])
    df_featimp2.sort_values('importance',ascending=False,inplace=True)
    df_featimp2.reset_index(drop=True,inplace=True)
    dfother=pd.DataFrame({"feature":["Others"],"importance":[sum(df_featimp2.loc[4:,'importance'])]})
    df_features=pd.concat([df_featimp2.iloc[0:4,],dfother])
    return df_features


def getFeatImps(model,df_x,df_feats):
    
    t3=sns.barplot(y=df_feats['importance'],x=df_feats['feature'])
    t3.set_xticklabels(df_feats['feature'],rotation='14')
    img3=t3.get_figure()
    img3.savefig("featsBar.png",bbox_inches='tight')
    t3.remove()
    matplotlib.pyplot.close()

    

def getDistro(df_y):
    t9=sns.histplot(df_y)
    t9.set_xlabel("Label Values",fontsize=15)
    t9.set_ylabel("Counts",fontsize=15)
    img9=t9.get_figure()
    img9.savefig("histPlot.png",bbox_inches='tight')
    t9.remove()
    
    
def getLime_clf(df_x,df_y,model):
    import lime
    import lime.lime_tabular
    explainer = lime.lime_tabular.LimeTabularExplainer(df_x.values, feature_names=df_x.columns,class_names=list(df_y.unique()) ,discretize_continuous=True)
    explanation = explainer.explain_instance(df_x.iloc[0].values, model.predict_proba)
    feature_importance = explanation.as_list()
    feats=[]
    val=[]
    for item in feature_importance:
        toks=item[0].split(' ')
        for tok in toks:
            if tok in df_x.columns:
                feats.append(tok)
                break
        val.append(abs(item[1]))

    
def classifierReport(df_x,df_y,model,filename="Tesst.pdf"):
    
    import os
    preds=model.predict(df_x)
    predsio=model.predict_proba(df_x)
    
    ct=canvas.Canvas(filename)
    ct.drawString(150,200,"Testfile Content v3")
    
    ct.setFont(psfontname='Times-Roman',size=15)
    ct.drawString(290,822,'PCA',)
    image_name="graphPCA_pred.png"
    getPcaPlot(df_x,preds,"PCA Predictions",image_name)
    ct.drawImage(image_name,x=20,y=578,width=265,height=240)
    os.remove(image_name)
    image_name="graphPCA_act.png"
    pca=getPcaPlot(df_x,df_y,"PCA Actual",image_name)
    ct.drawImage(image_name,x=300,y=578,width=270,height=240)
    ct.setFont(psfontname='Times-Roman',size=14)
    os.remove(image_name)
    
    
    ct.setFont(psfontname='Times-Roman',size=13)
    evals= getEvals_clf(df_y,preds,predsio)
    ct.drawString(35,302,"Accuracy")
    ct.drawString(95,302,evals[0])
    ct.drawString(35,282,"Log Loss")
    ct.drawString(95,282,evals[1])
    ct.drawString(195,302,"Precision")
    ct.drawString(255,302,evals[2])
    ct.drawString(195,282,"Recall")
    ct.drawString(255,282,evals[3])
    model_type=str(model).split('(')[0]
    ct.drawString(385,302,"Model Type:")
    ct.drawString(385,282,model_type)
    
    getConfusionMatrix(df_y,preds)
    ct.setFont(psfontname='Times-Roman',size=15)
    ct.drawString(390,558,"Confusion Matrix")
    ct.drawImage("heatmapConfMatrix.png",x=315,y=325,height=225,width=250)
    os.remove("heatmapConfMatrix.png")
    

    getDistro(df_y)
    ct.setFont(psfontname='Times-Roman',size=15)
    ct.drawString(110,558,'Label Distribution',)
    ct.drawImage("histPlot.png",x=20,y=325,width=266,height=225)
    os.remove("histPlot.png")

    
    feats_df=standardFeatImps(pca,df_x)
    getFeatImps(model,df_x,feats_df)
    ct.drawString(116,248,'Feature Importance',)
    ct.drawImage("featsBar.png",x=20,y=14,width=250,height=225)
    os.remove("featsBar.png")
    
    
    getPermFeatImp(df_x, df_y, model)
    ct.drawString(346,248,"Permutation Feature Importance")
    ct.drawImage("barPlot.png",x=295,y=14,height=225,width=270)
    os.remove("barPlot.png")
    
    ct.save()
    

def getEvals_reg(df_y,preds):
    from sklearn import metrics
    evals=['','','','']
    evals[0] = str(round(metrics.mean_squared_error(df_y,preds),2))
    evals[1] = str(round(metrics.mean_absolute_error(df_y,preds),2))
    evals[2] = str(round(metrics.mean_absolute_percentage_error(df_y,preds),2))
    evals[3] = str(round(metrics.mean_squared_log_error(df_y,preds),2))
    return evals

def getPermFeatImp(df_x,df_y,model):
    from sklearn.inspection import permutation_importance
    result_test = permutation_importance(model, df_x, df_y, n_repeats=20, random_state=42, n_jobs=2)
    sorted_importances_idx_test = result_test.importances_mean.argsort()
    importances_test = pd.DataFrame(result_test.importances[sorted_importances_idx_test].T,columns=df_x.columns[sorted_importances_idx_test],)
    t9=sns.barplot(data=importances_test, orient='h')
    t9.set_xlabel("Importance",fontsize=15)
    t9.set_ylabel("Feature",fontsize=15)
    img9=t9.get_figure()
    img9.savefig("barPlot.png",bbox_inches='tight')
    t9.remove()


def getResidual(df_y,preds):
    residuals=df_y-preds
    t8=sns.scatterplot(y=residuals,x=df_y)
    t8.set_xlabel("Actual",fontsize=15)
    t8.set_ylabel("Residual",fontsize=15)
    img8=t8.get_figure()
    img8.savefig("scatterPlot.png",bbox_inches='tight')
    t8.remove()


def regressorReport(df_x,df_y,model,filename="TesstRef.pdf"):
    
    import os
    preds=model.predict(df_x)
    
    ct=canvas.Canvas(filename)
    
    ct.setFont(psfontname='Times-Roman',size=15)
    ct.drawString(290,822,'PCA',)
    image_name="graphPCA_pred.png"
    pca=getPcaPlot(df_x,preds,"PCA Predictions",image_name)
    ct.drawImage(image_name,x=20,y=578,width=270,height=240)
    os.remove(image_name)
    image_name="graphPCA_act.png"
    getPcaPlot(df_x,df_y,"PCA Actual",image_name)
    ct.drawImage(image_name,x=300,y=578,width=265,height=240)
    ct.setFont(psfontname='Times-Roman',size=14)
    os.remove(image_name)
    
    
    ct.setFont(psfontname='Times-Roman',size=13)
    evals= getEvals_reg(df_y,preds)
    ct.drawString(35,302,"Mean Squared Error")
    ct.drawString(155,302,evals[0])
    ct.drawString(35,282,"Mean Absolute Error")
    ct.drawString(155,282,evals[1])
    ct.drawString(255,302,"Mean Squared Log Error")
    ct.drawString(435,302,evals[3])
    ct.drawString(255,282,"Mean Absolute Percentage Error")
    ct.drawString(435,282,evals[2])
    
    feats_df=standardFeatImps(pca,df_x)
    
    
    getFeatImps(model,df_x,feats_df)
    ct.setFont(psfontname='Times-Roman',size=15)
    ct.drawString(116,248,'Feature Importance',)
    ct.drawImage("featsBar.png",x=20,y=14,width=255,height=225)
    os.remove("featsBar.png")
    
    getDistro(df_y)
    ct.setFont(psfontname='Times-Roman',size=15)
    ct.drawString(110,558,'Label Distribution',)
    ct.drawImage("histPlot.png",x=20,y=325,width=266,height=225)
    os.remove("histPlot.png")
    
    getResidual(df_y,preds)
    ct.setFont(psfontname='Times-Roman',size=15)
    ct.drawString(390,558,"Residual Plot")
    ct.drawImage("scatterPlot.png",x=295,y=325,height=225,width=270)
    os.remove("scatterPlot.png")
    
    getPermFeatImp(df_x, df_y, model)
    ct.setFont(psfontname='Times-Roman',size=15)
    ct.drawString(346,248,"Permutation Feature Importance")
    ct.drawImage("barPlot.png",x=295,y=14,height=225,width=270)
    os.remove("barPlot.png")
    
    
    
    ct.save()
    
    
