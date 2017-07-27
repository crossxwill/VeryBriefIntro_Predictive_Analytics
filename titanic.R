### 1. decide what you want to predict -----

### I want to predict the probability of surviving the titanic

### 2a. find the data -----

??titanic

library(earth)

data(etitanic)

?etitanic

### 2b. explore the data -----

summary(etitanic)

with(etitanic, table(survived))
with(etitanic, table(survived, pclass))

surv_count <- with(etitanic, table(survived))
surv_by_class_count <- with(etitanic, table(survived, pclass))

prop.table(surv_count)
prop.table(surv_by_class_count, margin=1)
prop.table(surv_by_class_count, margin=2)

with(etitanic, plot(pclass, as.factor(survived), xlab="pclass"
                    , ylab="Survival Rate"))

with(etitanic, plot(sex, as.factor(survived), xlab="sex"
                    , ylab="Survival Rate"))

### 3. transform the data ----

etitanic$survived_d <- as.factor(ifelse(etitanic$survived == 1, 'Alive', 'Dead'))

etitanic$survived_d <- relevel(etitanic$survived_d, ref='Dead')

summary(etitanic)

table(etitanic$survived)

etitanic2 <- subset(etitanic, select=-c(survived))

summary(etitanic2)

### 4. split the data between train and test -----

library(caret)

set.seed(1234)
training <- createDataPartition(etitanic2$survived_d, p = 0.7, list=FALSE)

trainData <- etitanic2[training,]
testData <- etitanic2[-training,]

### 5. choose a performance metric --- 

fitControl <- trainControl(method="cv", classProbs = TRUE, summaryFunction = twoClassSummary)
perfmetric <- "ROC"

### 6. try many algos ----

## logistic regression
set.seed(1234)
logit.mod <- train(survived_d ~ ., data=trainData, trControl=fitControl, method="glm", metric=perfmetric)

## MARS
set.seed(1234)
earth.mod <- train(survived_d ~ ., data=trainData, trControl=fitControl, method="earth", metric=perfmetric)

## CART
set.seed(1234)
cart.mod <- train(survived_d ~ ., data=trainData, trControl=fitControl, method="rpart", metric=perfmetric)

## Random Forest
set.seed(1234)
rf.mod <- train(survived_d ~ ., data=trainData, trControl=fitControl, method="rf", metric=perfmetric)

## Gradient Boosting
set.seed(1234)
boosting.mod <- train(survived_d ~ ., data=trainData, trControl=fitControl, method="xgbTree", metric=perfmetric)

### 7. pick the best algo -----

preds_testData <- predict(list(logit=logit.mod, MARS=earth.mod, CART=cart.mod, RF=rf.mod, Boosting=boosting.mod), newdata=testData
                          ,type="prob")

str(preds_testData)

keepOnlyAlive <- function(df){
  keep <- df$Alive
  return(keep)
}

preds_testData_filt <- lapply(preds_testData, keepOnlyAlive)

preds_testData_df <- as.data.frame(preds_testData_filt)

str(preds_testData_df)

library(caTools)

allMods <- colAUC(preds_testData_df, testData$survived_d, plotROC = TRUE)

allMods

### 8. fit the final model using all the data ---

set.seed(1234)
final.boosting.mod <- train(survived_d ~ ., data=etitanic2, trControl=fitControl, method="xgbTree", metric=perfmetric)
varImp(final.boosting.mod)

set.seed(1234)
final.logit.mod <- train(survived_d ~ ., data=etitanic2, trControl=fitControl, method="glm", metric=perfmetric)
varImp(final.logit.mod)
summary(final.logit.mod$finalModel)