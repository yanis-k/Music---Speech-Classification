library(MLmetrics)
library(e1071)
library(scatterplot3d)
library(ROCR)
library(caret)
library(ROSE)

rm(list=ls())

# Load the data
alldata_raw = read.csv("dataset_SFB.csv")  #Windows Dir
temp <- c()
music <- c()
speech <- c()
for (i in 1:nrow(alldata_raw)) {
  if(alldata_raw[i, 22] == "music"){ music <- rbind(music, alldata_raw[i,]) }
  else { speech <- rbind(speech, alldata_raw[i,])}
}

#Training Set Segmentation
temp = rbind(temp, music[1:(nrow(music)/5),])
temp = rbind(temp, speech[1:(nrow(speech)/5),])
temp = rbind(temp, music[((nrow(music)/5) + 1):(2*(nrow(music)/5)), ])
temp = rbind(temp, speech[((nrow(speech)/5) + 1):(2*(nrow(speech)/5)), ])
temp = rbind(temp, music[(2*((nrow(music)/5) + 1)):(3*(nrow(music)/5)), ])
temp = rbind(temp, speech[(2*((nrow(speech)/5) + 1)):(3*(nrow(speech)/5)), ])
temp = rbind(temp, music[(3*((nrow(music)/5) + 1)):(4*(nrow(music)/5)), ])
temp = rbind(temp, speech[(3*((nrow(speech)/5) + 1)):(4*(nrow(speech)/5)), ])
temp = rbind(temp, music[(4*((nrow(music)/5) + 1)):(5*(nrow(music)/5)), ])
temp = rbind(temp, speech[(4*((nrow(speech)/5) + 1)):(5*(nrow(speech)/5)), ])

alldata_raw = rbind(temp, read.csv("dataset_SFB.csv"))
alldata_raw$rms <- NULL

#Preprocessing
alldata_raw = unique(alldata_raw) #delete duplicate entries
alldata_unique <- alldata_raw[complete.cases(alldata_raw), ]

class = alldata_unique[, 21]    #split data columns into features(cols 1-20) and class(col 21)
data_feats = alldata_unique[, -21]
cor(data_feats)

pca_model = prcomp(data_feats, center = TRUE, scale = TRUE) #PCA, normalized

summary(pca_model)

eigenvalues = pca_model$sdev^2
eigenvectors = pca_model$rotation
barplot(eigenvalues / sum(eigenvalues)) #Barplot of components' contribution


data_feats_pc <- as.data.frame(predict(pca_model, data_feats)[, 1:15]) #Keeping feats 1-15
data_feats_pc[, 16:20] <- 0 #Ignore feats 16-20
data_feats_rec = data.frame(t(t(as.matrix(data_feats_pc) %*%
                                 t(pca_model$rotation)) * pca_model$scale
                                  + pca_model$center)) #apply the transformation
data_feats_rec[, 16:20] <- NULL #Throw Away Components 16-20
data = cbind(data_feats_rec, class) #compiling final datasets

gammavalues = c(0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000) #Possible Gamma values for the SVM Model
costvalues = c(0.01, 0.1, 1, 5, 10, 100, 200, 500, 1000, 1500) #possible cost values

# Apply k-fold cross validation to find the best value for gamma
k = 6

# Split in k folds
data_size = nrow(data)
folds = split((1:data_size), ceiling(seq(data_size) * k / data_size))

accuracies <- c()

training_error <- c()
testing_error <- c()

for (gamma in gammavalues) {
  predictions <- data.frame()
  testsets <- data.frame()
  for(i in 1:k){
    # Select k-1 out of k folds for training and 1 for validation
    trainingset <- data[unlist(folds[-i]),]
    testset <- data[unlist(folds[i]),]
    
    # Train and apply the model
    svm_model = svm(class ~ ., kernel="radial", type="C-classification", data = trainingset, gamma = gamma)
    pred = predict(svm_model, testset[,-16])
    
    training_error = c(training_error, 1 - Accuracy(trainingset$class, pred))
    testing_error = c(testing_error, 1 - Accuracy(testset$class, pred))
    
    # Save predictions and testsets
    predictions <- rbind(predictions, as.data.frame(pred))
    testsets <- rbind(testsets, as.data.frame(testset[, 16]))
  }
  
  # Calculate the new accuracy and add it to the previous ones
  accuracies = c(accuracies, Accuracy(predictions, testsets))
}

# Find the best gamma value
opt_gamma = gammavalues[which.max(accuracies)]
print(paste0('Max Accuracy: ', max(accuracies), ' for gamma = ', opt_gamma))

#AVG Training and Testing Error per K-Fold Rotation
training_error = .colMeans(training_error, k, length(training_error) / k)
testing_error = .colMeans(testing_error, k, length(testing_error) / k)

#Plot Training and Testing Error
plot(training_error, type = "l", col="blue", ylim = c(0, 1), xlab = "Gamma", ylab = "Error", xaxt = "n")
axis(1, at = 1:length(gammavalues), labels = gammavalues)
lines(testing_error, col="red")
legend("right", c("Training Error", "Testing Error"), pch = c("-","-"), col = c("blue", "red"))

# Simulate for Optimal Cost value
# Split in k folds
folds = split((1:data_size), ceiling(seq(data_size) * k / data_size))

accuracies <- c()

training_error <- c()
testing_error <- c()

for (cost in costvalues) {
  predictions <- data.frame()
  testsets <- data.frame()
  for(i in 1:k){
    # Select k-1 out of k folds for training and 1 for validation
    trainingset <- data[unlist(folds[-i]),]
    testset <- data[unlist(folds[i]),]
    
    # Train and apply the model
    svm_model = svm(class ~ ., kernel="radial", type="C-classification", data = trainingset, cost=cost, gamma = opt_gamma)
    pred = predict(svm_model, testset[,-16])
    
    training_error = c(training_error, 1 - Accuracy(trainingset$class, pred))
    testing_error = c(testing_error, 1 - Accuracy(testset$class, pred))
    
    # Save predictions and testsets
    predictions <- rbind(predictions, as.data.frame(pred))
    testsets <- rbind(testsets, as.data.frame(testset[, 16]))
  }
  
  # Calculate the new accuracy and add it to the previous ones
  accuracies = c(accuracies, Accuracy(predictions, testsets))
}

# Find the best cost value
opt_cost = costvalues[which.max(accuracies)]
print(paste0('Max Accuracy: ', max(accuracies), ' for cost = ', opt_cost))

#AVG Training and Testing Error per K-Fold Rotation
training_error = .colMeans(training_error, k, length(training_error) / k)
testing_error = .colMeans(testing_error, k, length(testing_error) / k)

#Plot Training and Testing Error
plot(training_error, type = "l", col="blue", ylim = c(0, 1), xlab = "Cost", ylab = "Error", xaxt = "n")
axis(1, at = 1:length(costvalues), labels = costvalues)
lines(testing_error, col="red")
legend("right", c("Training Error", "Testing Error"), pch = c("-","-"), col = c("blue", "red"))

#Run the model again with optimal parameters to gain evaluation metrics
predictions <- data.frame()
testsets <- data.frame()
for(i in 1:k){
  
  trainingset <- data[unlist(folds[-i]),]
  testset <- data[unlist(folds[i]),]
  
  svm_model = svm(class ~ ., kernel="radial", type="C-classification", data = trainingset, cost = opt_cost, gamma = opt_gamma)
  pred = predict(svm_model, testset[,-16], type="raw")
  
  predictions <- rbind(predictions, as.data.frame(pred))
  testsets <- rbind(testsets, as.data.frame(testset[, 16]))
}

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

