setwd("~/Swetha/SWE PERSONAL/YALE/ACADS/Big Data/HW")

install.packages("ROCR")

library(readxl)
library(ROCR)

eBay=read_excel("eBay.xlsx")
fix(eBay)
names(eBay)
eBay$Category=factor(eBay$Category)
eBay$currency=factor(eBay$currency)
eBay$endDay=factor(eBay$endDay)
eBay$Duration=factor(eBay$Duration)
eBay$Competitive=factor(eBay$Competitive)

contrasts(eBay$endDay)

#Dividing eBay data into training and validation dataset
nrow(eBay)
set.seed(1)
trainindex=sample(nrow(eBay),0.6*nrow(eBay))
train=eBay[trainindex, ]
valid=eBay[-trainindex, ]

#glm for logistic regression
LR.model=glm(Competitive~.,data=train,family=binomial)
summary(LR.model)
coef(LR.model)


#Predict on validation set
Probability = predict(LR.model,new=valid,type="response")
pred = rep("0",nrow(valid))
pred[Probability>=0.5]="1"
Actual.class=valid$Competitive
table(Actual.class,pred)

#Overall Error = 0.2446

# To generate the ROC (Receiver operating characteristic) curve: function "performance()" 

roc = performance(prediction(Probability,Actual.class),"tpr", "fpr")
plot(roc)

# To compute the AUC (area under curve) on validation set
AUC = performance(prediction(Probability,Actual.class),"auc")
AUC@y.values 
#0.8293

# To see the false positive rate (fpr) and true positive rate (tpr) for different cutoff values:

cutoffs = data.frame(cut=roc@alpha.values[[1]], fpr=roc@x.values[[1]], tpr=roc@y.values[[1]])
head(cutoffs)

############
#Without Closing Price

#Dividing eBay data into training and validation dataset
nrow(eBay)
set.seed(1)
trainindex=sample(nrow(eBay),0.6*nrow(eBay))
train=eBay[trainindex, ]
valid=eBay[-trainindex, ]

#glm for logistic regression
LR.model=glm(Competitive~.-ClosePrice,data=train,family=binomial)
summary(LR.model)
coef(LR.model)

#Predict on validation set
Probability = predict(LR.model,new=valid,type="response")
pred = rep("0",nrow(valid))
pred[Probability>=0.5]="1"
Actual.class=valid$Competitive
table(Actual.class,pred)

#Overall Error = 0.3662

# To generate the ROC curve: function "performance()" 

roc = performance(prediction(Probability,Actual.class),"tpr", "fpr")
plot(roc)

# To compute the AUC (area under curve) on validation set
AUC = performance(prediction(Probability,Actual.class),"auc")
AUC@y.values 
#0.6799

# To see the false positive rate (fpr) and true positive rate (tpr) for different cutoff values:

cutoffs = data.frame(cut=roc@alpha.values[[1]], fpr=roc@x.values[[1]], tpr=roc@y.values[[1]])
head(cutoffs)

###########################


