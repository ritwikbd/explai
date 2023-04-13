# explai
A reporting tool for Explainable AI
</br> Installation:
</br> pip install explai

</br>
It has 2 functions:
</br> classifierReport(df_x,df_y,model,filename) : For classification models
</br> regressorReport(df_x,df_y,model,filename) : For regression models

</br> df_x and df_y are the x and y of your validation set respectively
</br> model is the model you have trained
</br> filename is the name of the file you want. (Optional - default is explaiReport.pdf)
