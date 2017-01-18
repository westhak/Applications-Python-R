##Tree Models

setwd("~/Swetha/SWE PERSONAL/YALE/ACADS/Big Data/HW")

install.packages('readxl')
library(readxl)
library(tree)

eBay=read_excel("eBay.xlsx")
fix(eBay)
names(eBay)

##Set as categorical variables

eBay$Category=factor(eBay$Category)
eBay$currency=factor(eBay$currency)
eBay$endDay=factor(eBay$endDay)
eBay$Duration=factor(eBay$Duration)
eBay$Competitive=factor(eBay$Competitive)

#Dividing eBay data into training and validation dataset
nrow(eBay)
set.seed(1)
trainindex=sample(nrow(eBay),0.8*nrow(eBay))
train=eBay[trainindex, ]
valid=eBay[-trainindex, ]

eBaytree=tree(Competitive~.,data=train)
summary(eBaytree)

plot(eBaytree, type = "uniform")
text(eBaytree, pretty = 0)

#Optimize nodes by plotting against RMS_Error
cv=cv.tree(eBaytree)
Nodes=cv$size
Nodes
RMS_Error=cv$dev
plot(Nodes,RMS_Error)
title('Cross validated RMS error as function of #terminal nodes')


#Prune Regression Tree by evaluating graph
eBaytree2 = prune.tree(eBaytree, best = 7)
summary(eBaytree2)

#PLot pruned regression tree
plot(eBaytree2, type = "uniform")
text(eBaytree2, pretty = 0)
# Two most important variables: Open Price, Close Price

#Overall Error rate
Tree.pred = predict(eBaytree2, new = valid, type="class")
Actual.class=valid$Competitive
table(Actual.class,Tree.pred)

overall.error=sum(Actual.class!=Tree.pred)/nrow(valid)
overall.error #0.1924

######################################
#Without close price
set.seed(1)
trainindex=sample(nrow(eBay),0.8*nrow(eBay))
train=eBay[trainindex, ]
valid=eBay[-trainindex, ]

eBaytree=tree(Competitive~.-ClosePrice,data=train)
summary(eBaytree)

plot(eBaytree, type = "uniform")
text(eBaytree, pretty = 0)

#Optimize nodes by plotting against RMS_Error
cv=cv.tree(eBaytree)
Nodes=cv$size
Nodes
RMS_Error=cv$dev
plot(Nodes,RMS_Error)
title('Cross validated RMS error as function of #terminal nodes')

#Prune Regression Tree by evaluating graph
eBaytree2 = prune.tree(eBaytree, best = 4)
summary(eBaytree2)

#PLot pruned regression tree
plot(eBaytree2, type = "uniform")
text(eBaytree2,pretty = 0)
# Two most important variables: Open Price, Close Price

#Overall Error rate
Tree.pred = predict(eBaytree2, new = valid, type="class")
Tree.pred
Actual.class=valid$Competitive
table(Actual.class,Tree.pred)

overall.error=sum(Actual.class!=Tree.pred)/nrow(valid)
overall.error #0.2734
####################VALIDATE

#Predict for given values
test=read_excel("eBay.xlsx", sheet=2)
test$Category=factor(test$Category)
test$currency=factor(test$currency)
test$endDay=factor(test$endDay)
test$Duration=factor(test$Duration)

Tree.pred.test = predict(eBaytree2, new =test , type="class")
Tree.pred.test
#Ans: 1 
####################
