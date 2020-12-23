# Library packages
library(fastDummies)
library(DMwR)
library(MLmetrics)
library(randomForest)
library(e1071)
library(Rcpp)
library(RSNNS)
library(ROCR)
library(pROC)
library(ROSE)


# Import data
online_shopper <- read.csv("online_shoppers_intention.csv", header = T)


###### exploratory data analysis
# stacked bat plots for VisitorType
plot.df <- data.frame(matrix(0, 6, 3))
colnames(plot.df) <- c("VistiorType", "Revenue", "Percentage")
plot.df$VistiorType <- rep(c("Returning_Visitor", "New_Visitor", "Other"), 2)
plot.df$Revenue <- c(TRUE, TRUE, TRUE, FALSE, FALSE, FALSE)
plot.df$Percentage <- c(0.77, 0.221, 0.009, 0.871, 0.122, 0.007)

ggplot( data = plot.df) + geom_bar(aes(y = Percentage, x = Revenue, fill = VistiorType),
                                   stat="identity", width = 0.5)+
  theme(legend.position="bottom", legend.direction="horizontal", legend.title = element_blank(),
        plot.title = element_text(hjust = 0.5)) + 
  ggtitle("VistorType Percentage in Different Sopping Behaviour Groups")

# boxplot for PageValues
ggplot(online_shopper, aes(x=PageValues, fill=Revenue)) + 
  geom_boxplot() + scale_x_continuous(limits = c(0, 75)) + coord_flip() +
  ggtitle("PageValues in Different Sopping Behaviour Groups") + 
  theme(plot.title = element_text(hjust = 0.5))

##### Data Processing for models
# label variables as factor
online_shopper$Revenue <- as.factor(online_shopper$Revenue)
# dummy variables
col_names <- c("Month", "OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType", "Weekend")
data_dummy <- dummy_cols(online_shopper, select_columns = col_names,
                         remove_selected_columns = TRUE)
# Split training and test data
n <- dim(data_dummy)[1]
p <- dim(data_dummy)[2]
set.seed(1234)
train_index <- sample(1:n, 0.7*n)
train_data <- data_dummy[train_index,]
test_data <- data_dummy[-train_index,]
# Using Somte method to balance data
train_data_b2 <- SMOTE(Revenue~., train_data, perc.over = 200, perc.under =200) #F=5336,T=4002

##### Random Forest
# hyper parameter tuning
rf.tune <- tune(randomForest,Revenue ~ ., data = train_data,mtry=8,ranges = list(ntree=c(250,500,750,1000)))
# mtry=8,ntree= 500
rf.fit <- randomForest(Revenue ~ ., data = train_data, mtry = 8, ntree=500,importance = T)
rf.pred <- predict(rf.fit, newdata = test_data)
rf.prob <- predict(rf.fit, newdata = test_data, type = 'prob')
#p1 = roc.curve(test_data$Revenue,rf.prob[,2], main = 'ROC of Random Forest')
mean(test_data$Revenue == rf.pred) #0.9083
auc <- roc(test_data$Revenue, rf.prob[,2])
print(auc) #0.9315
F1_Score(y_pred = rf.pred, y_true = test_data$Revenue, positive = "TRUE") #0.6412
F1_Score(y_pred = rf.pred, y_true = test_data$Revenue, positive = "FALSE") #0.94

# Variable importance

imp <- importance(rf.fit,type=1)
variable <- rownames(imp)
varIMP <- data.frame(variable,imp)
varIMP_sort <- varIMP[order(-varIMP$MeanDecreaseAccuracy),]
ggplot(varIMP_sort[1:8,], aes(x = variable, y = MeanDecreaseAccuracy))+ 
  geom_bar(position = "dodge", stat = "identity",fill = "skyblue", colour = "black")+
  coord_flip()

# balanced 5:5
rf.fitb2 <- randomForest(Revenue ~ ., data = train_data_b2, mtry = 8, ntree=500,importance = T)
rf.pred.b2 <- predict(rf.fitb2, newdata = test_data)
rf.prob.b2 <- predict(rf.fitb2, newdata = test_data, type = 'prob')
#p1 = roc.curve(test_data$Revenue,rf.prob[,2], main = 'ROC of Random Forest')
mean(test_data$Revenue == rf.pred.b2) #0.9075
auc <- roc(test_data$Revenue, rf.prob.b2[,2])
print(auc) #0.9328
F1_Score(y_pred = rf.pred.b2, y_true = test_data$Revenue, positive = "TRUE") #0.7121
F1_Score(y_pred = rf.pred.b2, y_true = test_data$Revenue, positive = "FALSE") #0.9449


##### logistic regression
index.R <- grep("Revenue", colnames(data_dummy))
# split X variables and outcome Y
train_data_X <- train_data[,-index.R]
test_data_X <- test_data[,-index.R]
mid <- train_data
mid_test <- test_data

mid$Revenue <- as.factor(mid$Revenue)
mid_test$Revenue <- as.factor(mid_test$Revenue)
# select lambda by cross validation
cv.logi <- cv.glmnet(data.matrix(train_data_X), data.matrix(mid$Revenue), family = "binomial",
                     alpha = 1) #lambda=0.00168

best.logi <- glmnet(data.matrix(train_data_X), data.matrix(mid$Revenue), family = "binomial",
                    alpha = 1, lambda = cv.logi$lambda.min)

prob <- predict(best.logi, newx = data.matrix(test_data_X), type = "response")
pred <- predict(best.logi, newx = data.matrix(test_data_X), type = "class")
mean(pred==mid_test$Revenue) # 0.890713
# ROC Curve
p_logi <- roc.curve(mid_test$Revenue, predicted = prob)
# F1 score
F1_Score(y_pred = pred, y_true = mid_test$Revenue, positive = T) #0.5367
F1_Score(y_pred = pred, y_true = mid_test$Revenue, positive = F) #0.9381

# Balanced data
# split X variables and outcome Y
train_data_X.b <- train_data_b2[,-index.R]
mid.b <- train_data_b2
# select lambda by cross validation
cv.logi.b <- cv.glmnet(data.matrix(train_data_X.b), data.matrix(mid.b$Revenue), family = "binomial",
                     alpha = 1) # lambda=0.00454

best.logi.b <- glmnet(data.matrix(train_data_X.b), data.matrix(mid.b$Revenue), family = "binomial",
                    alpha = 1, lambda = cv.logi.b$lambda.min)

prob <- predict(best.logi.b, newx = data.matrix(test_data_X), type = "response")
pred <- predict(best.logi.b, newx = data.matrix(test_data_X), type = "class")
mean(pred==mid_test$Revenue) # 0.8821303
# F1 score
F1_Score(y_pred = pred, y_true = mid_test$Revenue, positive = T) #0.6355
F1_Score(y_pred = pred, y_true = mid_test$Revenue, positive = F) #0.9297

# MLP and SVM code on python. The two model will cosume much time in R.
# We write train.data and test.data in case all model based on the same data.