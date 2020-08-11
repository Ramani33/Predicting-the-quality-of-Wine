# STAT 515 FINAL PROJECT #
# GROUP MEMBERS : ANANYA, RAMANI, AVINASH #

----------------------------------------------------------------------------
### SETUP ###
# libraries
install.packages("e1071")
install.packages("GGally")
install.packages("party")
install.packages("ROCR")
install.packages("rpart.plot")
install.packages("randomForest")
library(dplyr)
library(ggplot2)
library(caTools)
library(caret)
library(GGally)
library(e1071)
library('party')
library('rpart')
library('rpart.plot')
library('caret')
library('ROCR')
library(tidyverse)
library(randomForest)
library(e1071)
-----------------------------------------------------------------------------

### Exploratory Data Analysis ###
  
#importing data set
data <- read.csv("~/Downloads/winequality.csv")
summary(data)
str(data)
nrow(data)
ncol(data)
sum(is.na(data))
for(i in 1:ncol(data)){
  data[is.na(data[,i]), i] <- mean(data[,i], na.rm = TRUE)
}
sum(is.na(data))

# checking ratio of different labels in target feature
prop.table(table(data$quality))
Data = data %>%
  mutate(quality_bin = as.factor(ifelse(quality <= 5, 0,1))) %>%
  select(-quality)
t = round(prop.table(table(data$quality_bin))*100,2)

# Exploring Predictors Visually
Data %>%
  ggplot(aes(x = as.factor(quality_bin), y = fixed.acidity, color = quality_bin)) +
  geom_boxplot(outlier.color = "blue", notch = FALSE) + 
  labs(title = paste0("Boxplot of feature: Fixed Acidity")) + ylab("Fixed Acidity") + xlab("Quality (1 = good, 2 = bad)") +
  theme(legend.position = "none", axis.title.x = element_blank()) + 
  theme_minimal()

Data %>%
  ggplot(aes(x = as.factor(quality_bin), y = volatile.acidity, color = quality_bin)) +
  geom_boxplot(outlier.color = "blue", notch = FALSE) + 
  labs(title = paste0("Boxplot of feature: Volatile acidity")) + ylab("Volatile Acidity") + xlab("Quality (1 = good, 2 = bad)") +
  theme(legend.position = "none", axis.title.x = element_blank()) + 
  theme_minimal()
### We have multiple features that are continuous and can plot them similarly

# plots continuous feature in boxplot categorized on the quality_bin feature labels from winedata 
# param feat Feature name (string) to be plotted
boxplot_viz = function(feat){
Data %>%
    ggplot(aes_string(x = as.factor('quality_bin'), y = feat, color = 'quality_bin')) +
    geom_boxplot(outlier.color = "purple", notch = FALSE) +
    labs(title = paste0("Boxplot of feature: ", feat)) + ylab(feat) + xlab("Quality (1 = good, 2 = bad)") + 
    theme(legend.position = "none", axis.title.x = element_blank()) + 
    theme_minimal()
}
boxplot_viz('volatile.acidity')
for (i in names(Data %>% select(-'quality_bin'))){
  print(boxplot_viz(i))
}

# correlation plot 
Data %>% 
  ggcorr(method = c('complete.obs','pearson'), 
         nbreaks = 6, digits = 3,label = TRUE, label_size = 3, 
         label_color = "black", label_round = 2, legend.size = 9, hjust = 0.8, size = 3, layout.exp = 1.5)


# check the Distribution of the wine quality
hist(data$quality, main= "Wine Quality", col= "pink")


# Scatterplot showing the difference between white and red wine.
df<-data[data$density<1.01,]
ggplot(df, aes(x=density, y=alcohol, color=type)) + 
  geom_point(size=3, alpha=0.6)

------------------------------------------------------------------------------
### FITTING MODEL ###
  
# Splitting the data
set.seed(123)
split = sample.split(Data$quality_bin, SplitRatio = 0.80)
training.set = subset(Data, split == TRUE)
test.set = subset(Data, split == FALSE)


# Data balance checking
prop.table(table(training.set$quality_bin))
prop.table(table(test.set$quality_bin))


# Fitting Logistic Regression classification model on our dataset
lr_model = glm(quality_bin ~ ., data = training.set, family = 'binomial')
#Summarize the model
summary(lr_model)

newdata = data.frame(hp=120, wt=2.8)
# Imortance of Features
# We plotted the variables with the lowest p values/highest absolute z value.
p = varImp(lr_model) %>% data.frame() 
p = p %>% mutate(Features = rownames(p)) %>% arrange(desc(Overall)) %>% mutate(Features = tolower(Features))

p %>% ggplot(aes(x = reorder(Features, Overall), y = Overall)) + geom_col(width = .50, fill = 'orange') + coord_flip() + 
  labs(title = "Importance of Features", subtitle = "Based on the value of individual z score") +
  xlab("Features") + ylab("Abs. Z Score") + 
  theme_minimal()


# Model Performance
pred = as.data.frame(predict(lr_model, type = "response", newdata = test.set)) %>% 
  structure( names = c("pred_prob")) %>%
  mutate(pred_cat = as.factor(ifelse(pred_prob > 0.5, "1", "0"))) %>% 
  mutate(actual_cat = test.set$quality_bin)
t = confusionMatrix(pred$pred_cat, pred$actual_cat, positive = "1")
t

#Split into Train and Test sets for Trees
wdata = data %>%
  mutate(quality_bin = as.factor(ifelse(quality <= 5, 0,1))) %>%
  select(-quality)
set.seed(100)
train <- sample(nrow(wdata), 0.7*nrow(data), replace = FALSE)
TrainSet <- wdata[train,]
TestSet <- wdata[-train,]
summary(TrainSet)
summary(TestSet)

#Decision Tree Classifier
dectree = train(quality_bin ~ ., data = TrainSet, method = "rpart")
print(dectree)
predict = predict(dectree, newdata = TestSet)
table(predict, TestSet$quality_bin)
mean(predict == TestSet$quality_bin)

### Random Forest Classifier
# choosing the right mtry
a=c()
i=5
for (i in 3:8) {
  randf <- randomForest(quality_bin ~ ., data = TrainSet, ntree = 500, mtry = i, importance = TRUE)
  pred <- predict(randf, TestSet, type = "class")
  a[i-2] = mean(pred == TestSet$quality_bin)
}

a

plot(3:8,a)

#Fitting the model for the best mtry
randf <- randomForest(quality_bin ~ ., data = TrainSet, ntree = 500, mtry = 3, importance = TRUE)
summary(randf)
print(randf)

#CHECKING THE MODEL PERFORMANCE
pred <- predict(randf, TestSet, type = "class")
mean(pred == TestSet$quality_bin)                    
varImpPlot(randf)

----------------------------------------------------------------------------


