# explai
A reporting tool for Explainable AI


## Installation:

```
pip install explai
```

## Two main functions are:
* **classification task**: `classifierReport(df_x,df_y,model,filename)`
* **regression task**: `regressorReport(df_x,df_y,model,filename)`

</br> df_x and df_y are the x and y of your validation set respectively
</br> model is the model you have trained
</br> filename is the name of the file you want. (Optional - default is explaiReport.pdf)
