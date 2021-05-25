#SECTION 1: Introduction
#Install packages and download libraries necessary to download data and analyze
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(dslabs)) install.packages("dslabs")
if(!require(dplyr)) install.packages("dplyr")
if(!require(ggplot2)) install.packages("ggplot2")
if(!require(rpart)) install.packages("rpart")
if(!require(randomForest)) install.packages("randomForest")
if(!require(gam)) install.packages("gam")
if(!require(Rborist)) install.packages("Rborist")
if(!require(readr)) install.packages("readr")
if(!require(knitr)) install.packages("knitr")
if(!require(repmis)) install.packages("repmis")

library(tidyverse)
library(caret)
library(data.table)
library(dslabs)
library(dplyr)
library(ggplot2)
library(rpart)
library(randomForest)
library(gam)
library(Rborist)
library(readr)
library(knitr)
library(repmis)

#download data from Github repository
grad_admissions<-read_csv("https://raw.githubusercontent.com/ghansonjr/USA-Grad-School-Admission/main/College_admission%202.csv")

set.seed(1,sample.kind="Rounding")

#SECTION 2: DATA EXPLORATION
dim(grad_admissions) #400 observations of 7 variables

mean(grad_admissions$admit) #gives the overall admittance rate across the data set.

#Analyze the spread of the gre scores in the grad_admissions data set
grad_admissions%>%ggplot(aes(gre))+geom_histogram(binwidth=60)+ggtitle("GRE Scores")
summary(grad_admissions$gre) #gives min, max, median, mean, and quartiles)
grad_admissions%>%group_by(gre)%>%summarize(AdmitRate=mean(admit))%>%ggplot(aes(x=gre,y=AdmitRate))+geom_point()+geom_smooth(method="lm")+ggtitle("GRE Score vs. Admittance Rate")#Dot plot showing relationship between gre scores and admittance rate

#Analyze the spread of the GPAs in the grad_admissions dataset
grad_admissions%>%ggplot(aes(gpa))+geom_histogram(binwidth=.25)+ggtitle("GPAs")
summary(grad_admissions$gpa) #gives min, max, median, mean, and quartiles)
#round gpa to nearest .5 then find admit rate by band
grad_admissions%>%mutate(gpa,gpa=round(gpa/.5)*.5)%>%group_by(gpa)%>%summarize(AdmitRate=mean(admit))

#Calculate correlation between gpa and GRE scores (display visually)
cor(grad_admissions$gre,grad_admissions$gpa)
grad_admissions%>%ggplot(aes(gpa,gre))+geom_point()+ggtitle("GPA vs. GRE")

#Analyze school rank effect
grad_admissions%>%group_by(rank)%>%summarize(n=n(),AdmitRate=mean(admit))

#Analyze SES effect
grad_admissions%>%group_by(ses)%>%summarize(n=n(),AdmitRate=mean(admit),AvgGPA=mean(gpa),AvgGRE=mean(gre))

#Analyze Gender effect
grad_admissions%>%group_by(Gender_Male)%>%summarize(n=n(),AdmitRate=mean(admit),AvgGPA=mean(gpa),AvgGRE=mean(gre))

#Analyze Race effect
grad_admissions%>%group_by(Race)%>%summarize(n=n(),AdmitRate=mean(admit),AvgGPA=mean(gpa),AvgGRE=mean(gre))

#Pull demographic variables from data set (remove ses, Gender, and Race)
grad_admissions_var<-subset(grad_admissions,select=-c(ses,Gender_Male,Race))
grad_admissions_var_x<-subset(grad_admissions_var,select=-c(admit))
grad_admissions_var_y<-as.factor(grad_admissions_var$admit)
#CREATE TRAIN AND TEST SETS FROM grad admissions set using 75% for the train set
test_index<-createDataPartition(y=grad_admissions_var$admit,times=1,p=0.25,list=FALSE)
train_admissions<-grad_admissions_var[-test_index,]
test_admissions<-grad_admissions_var[test_index,]

#Create x and y data frames from each set (y=admit column, x=other variables)
train_admissions_y<-as.factor(train_admissions$admit)
train_admissions_x<-subset(train_admissions,select=-c(admit))

test_admissions_y<-as.factor(test_admissions$admit)
test_admissions_x<-subset(test_admissions,select=-c(admit))

#CREATE AND ASSESS VARIOUS MODELS

#Using Overall Average
set.seed(1,sample.kind="Default")

mean_preds<-sample(c("1","0"),100,replace=TRUE,prob=c(mean(grad_admissions$admit),1-(mean(grad_admissions$admit))))
mean(mean_preds==test_admissions_y)

Result<-tibble(Method="Overall Average Guess",Accuracy=mean(mean_preds==test_admissions_y))

#GLM - logistic regression
train_glm <- train(train_admissions_x, train_admissions_y, method = "glm")
glm_preds <- predict(train_glm, test_admissions_x)
mean(glm_preds == test_admissions_y)
Result_glm<-bind_rows(Result,tibble(Method="GLM",Accuracy=mean(glm_preds==test_admissions_y)))
Result<-bind_rows(Result,tibble(Method="GLM",Accuracy=mean(glm_preds==test_admissions_y)))

#lda
train_lda <- train(train_admissions_x, train_admissions_y, method = "lda")
lda_preds <- predict(train_lda, test_admissions_x)
mean(lda_preds == test_admissions_y)

Result<-bind_rows(Result,tibble(Method="LDA",Accuracy=mean(lda_preds==test_admissions_y)))
                  
#qda
train_qda <- train(train_admissions_x, train_admissions_y, method = "qda")
qda_preds <- predict(train_qda, test_admissions_x)
mean(qda_preds == test_admissions_y)

Result<-bind_rows(Result,tibble(Method="QDA",Accuracy=mean(qda_preds==test_admissions_y)))

#loess
train_loess <- train(train_admissions_x, train_admissions_y, method = "gamLoess")
loess_preds <- predict(train_loess, test_admissions_x)
mean(loess_preds == test_admissions_y)

Result<-bind_rows(Result,tibble(Method="Loess",Accuracy=mean(loess_preds==test_admissions_y)))

#knn - Next nearest neighbors
tuning<-data.frame(k=seq(3,15,2))
train_knn<-train(train_admissions_x,train_admissions_y,method="knn",tuneGrid=tuning)
train_knn$bestTune #Displays Optimal # of nearest neighbors
knn_preds<-predict(train_knn,test_admissions_x)
mean(knn_preds==test_admissions_y)

Result<-bind_rows(Result,tibble(Method="KNN",Accuracy=mean(knn_preds==test_admissions_y)))

#Random Forest
train_rf<-train(train_admissions_x,train_admissions_y,method="rf")
rf_preds<-predict(train_rf,test_admissions_x)
mean(rf_preds==test_admissions_y)

Result_rf<-bind_rows(Result,tibble(Method="Random Forest",Accuracy=mean(rf_preds==test_admissions_y)))

Result<-bind_rows(Result,tibble(Method="Random Forest",Accuracy=mean(rf_preds==test_admissions_y)))

#Ensemble - average of the other methods determines prediction
ensemble_model<-cbind(glm=glm_preds=="1",lda=lda_preds=="1",qda=qda_preds=="1",loess=loess_preds=="1",knn=knn_preds=="1",rf=rf_preds=="1")
ensemble_preds<-ifelse(rowMeans(ensemble_model)>0.5,"1","0")
mean(ensemble_preds==test_admissions_y)

Result<-bind_rows(Result,tibble(Method="Ensemble",Accuracy=mean(ensemble_preds==test_admissions_y)))

#SECTION 3: RESULTS
#Print Result - Table of all of the accuracies from the various methods we have tried
Result

#Best Method
optimal_method<-Result$Method[which.max(Result$Accuracy)]
optimal_method

#Run the best method on the entire data set to get an accuracy for running it on a larger data set (this will likely be inflated due to overtraining as 75% of the data in the set was used to train this algorithm)
qda_preds_final<-predict(train_qda,grad_admissions_var_x)
mean(qda_preds_final==grad_admissions_var_y) #Final Accuracy