library(MLmetrics)
library(e1071)
library(scatterplot3d)
library(ROCR)
library(caret)
library(ROSE)

rm(list=ls())

# Load the data
alldata_raw_train = read.csv("dataset_STI.csv")  #Windows Dir
alldata_raw_test = read.csv("dataset_STI.csv")  #Windows Dir

alldata_raw_train$rms_mean <- NULL
alldata_raw_test$rms_mean <- NULL

alldata_raw_train$rms_std <- NULL
alldata_raw_test$rms_std <- NULL

#Preprocessing
alldata_unique_train = unique(na.omit(alldata_raw_train)) #delete duplicate and NA entries
alldata_unique_test = unique(na.omit(alldata_raw_test))

class_train = alldata_unique_train[, 41]    #split training data columns into features(cols 1-20) and class(col 21)
data_feats_train = alldata_unique_train[, -41]

class_test = alldata_unique_test[, 41]    #split testing data columns into features(cols 1-20) and class(col 21)
data_feats_test = alldata_unique_test[, -41]

pca_model_train = prcomp(data_feats_train, center = TRUE, scale = TRUE) #PCA, normalized
pca_model_test = prcomp(data_feats_test, center = TRUE, scale = TRUE) #PCA, normalized

summary(pca_model_train)
summary(pca_model_test)

eigenvalues_train = pca_model_train$sdev^2
eigenvectors_train = pca_model_train$rotation
barplot(eigenvalues_train / sum(eigenvalues_train)) #Barplot of components' contribution

data_feats_pc_train <- as.data.frame(predict(pca_model_train, data_feats_train)[, 1:30]) #Keeping feats 1-15
data_feats_pc_train[, 31:40] <- 0 #Ignore feats 16-20
data_feats_rec_train = data.frame(t(t(as.matrix(data_feats_pc_train) %*%
                                 t(pca_model_train$rotation)) * pca_model_train$scale
                                  + pca_model_train$center)) #apply the transformation
data_feats_rec_train[, 31:40] <- NULL #Throw Away Components 17-21
data_train = cbind(data_feats_rec_train, class_train) #compiling final training dataset

eigenvalues_test = pca_model_test$sdev^2
eigenvectors_test = pca_model_test$rotation
barplot(eigenvalues_test / sum(eigenvalues_test)) #Barplot of components' contribution

data_feats_pc_test <- as.data.frame(predict(pca_model_test, data_feats_test)[, 1:30]) #Keeping feats 1-15
data_feats_pc_test[, 31:40] <- 0 #Ignore feats 17-21
data_feats_rec_test = data.frame(t(t(as.matrix(data_feats_pc_test) %*%
                                        t(pca_model_test$rotation)) * pca_model_test$scale
                                    + pca_model_test$center)) #apply the transformation
data_feats_rec_test[, 31:40] <- NULL #Throw Away Components 17-21
data_test = cbind(data_feats_rec_test, class_test) #compiling final testing dataset

gammavalues = c(0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000) #Possible Gamma values for the SVM Model
costvalues = c(0.01, 0.1, 1, 5, 10, 100, 200, 500, 1000, 1500) #possible cost values

accuracies <- c()

training_error <- c()
testing_error <- c()

for (gamma in gammavalues) {
  predictions <- data.frame()
  testsets <- data.frame()

  # Train and apply the model
  svm_model = svm(class_train ~ ., kernel="radial", type="C-classification", data = data_train, gamma = gamma)
  pred = predict(svm_model, data_test[,-31])
    
  training_error = c(training_error, 1 - Accuracy(data_train$class, pred))
  testing_error = c(testing_error, 1 - Accuracy(data_test$class, pred))
    
  # Save predictions and testsets
  predictions <- rbind(predictions, as.data.frame(pred))
  testsets <- rbind(testsets, as.data.frame(data_test[, 31]))
  
  # Calculate the new accuracy and add it to the previous ones
  accuracies = c(accuracies, Accuracy(predictions, testsets))
}

# Find the best gamma value
opt_gamma = gammavalues[which.max(accuracies)]
print(paste0('Max Accuracy: ', max(accuracies), ' for gamma = ', opt_gamma))

#Plot Training and Testing Error
plot(training_error, type = "l", col="blue", ylim = c(0, 1), xlab = "Gamma", ylab = "Error", xaxt = "n")
axis(1, at = 1:length(gammavalues), labels = gammavalues)
lines(testing_error, col="red")
legend("right", c("Training Error", "Testing Error"), pch = c("-","-"), col = c("blue", "red"))

# Simulate for Optimal Cost value
accuracies <- c()

training_error <- c()
testing_error <- c()

for (cost in costvalues) {
  predictions <- data.frame()
  testsets <- data.frame()
  
  # Train and apply the model
  svm_model = svm(class_train ~ ., kernel="radial", type="C-classification", data = data_train, cost = cost, gamma = opt_gamma)
  pred = predict(svm_model, data_test[,-31])
  
  training_error = c(training_error, 1 - Accuracy(data_train$class, pred))
  testing_error = c(testing_error, 1 - Accuracy(data_test$class, pred))
  
  # Save predictions and testsets
  predictions <- rbind(predictions, as.data.frame(pred))
  testsets <- rbind(testsets, as.data.frame(data_test[, 31]))
  
  # Calculate the new accuracy and add it to the previous ones
  accuracies = c(accuracies, Accuracy(predictions, testsets))
}

# Find the best cost value
opt_cost = costvalues[which.max(accuracies)]
print(paste0('Max Accuracy: ', max(accuracies), ' for cost = ', opt_cost))

#Plot Training and Testing Error
plot(training_error, type = "l", col="blue", ylim = c(0, 1), xlab = "Cost", ylab = "Error", xaxt = "n")
axis(1, at = 1:length(costvalues), labels = costvalues)
lines(testing_error, col="red")
legend("right", c("Training Error", "Testing Error"), pch = c("-","-"), col = c("blue", "red"))

#Run the model again with optimal parameters to gain evaluation metrics
predictions <- data.frame()
testsets <- data.frame()

svm_model = svm(class_train ~ ., kernel="radial", type="C-classification", data = data_train, cost = opt_cost, gamma = opt_gamma)
pred = predict(svm_model, data_test[,-31], type="raw")
  
predictions <- rbind(predictions, as.data.frame(pred))
testsets <- rbind(testsets, as.data.frame(data_test[, 31]))

opt_accuracy = Accuracy(predictions, testsets) #Calculating Optimal Accuracy
preds.factor <- as.factor(predictions[,1])
tests.factor <- as.factor(testsets[,1])
confMat = confusionMatrix(preds.factor, tests.factor) #Calculating Confusion Matrix
opt_precision = Precision(tests.factor, preds.factor, "music") #Calculating Optimal Precision
opt_recall = Recall(tests.factor, preds.factor, "music") #Calculating Optimal Recall
opt_fmeasure = 2*(opt_precision*opt_recall)/(opt_precision + opt_recall) #Calculating F-measure accordingly

cat("Accuracy: ", opt_accuracy)
confMat
cat("Precision: ", opt_precision)
cat("Recall: ", opt_recall)
cat("F-measure: ", opt_fmeasure)

#Designing ROC Curve
roc.curve(preds.factor,tests.factor)
