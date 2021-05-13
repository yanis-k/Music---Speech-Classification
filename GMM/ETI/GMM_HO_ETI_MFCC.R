library(MLmetrics)
library(e1071)
library(scatterplot3d)
library(ROCR)
library(caret)
library(ROSE)
library(mclust)

rm(list=ls())

# Load the data
alldata_raw_train = read.csv("dataset_ETI.csv")  #Windows Dir
alldata_raw_test = read.csv("dataset_ETI.csv")  #Windows Dir

alldata_raw_train$rms_mean <- NULL
alldata_raw_test$rms_mean <- NULL
alldata_raw_train$rms_std <- NULL
alldata_raw_test$rms_std <- NULL
alldata_raw_train$rms_mcr <- NULL
alldata_raw_test$rms_mcr <- NULL
alldata_raw_train$rms_masd <- NULL
alldata_raw_test$rms_masd <- NULL
alldata_raw_train$rms_flat <- NULL
alldata_raw_test$rms_flat <- NULL

alldata_raw_train$zerocross_mean <- NULL
alldata_raw_test$zerocross_mean <- NULL
alldata_raw_train$zerocross_std <- NULL
alldata_raw_test$zerocross_std <- NULL
alldata_raw_train$zerocross_mcr <- NULL
alldata_raw_test$zerocross_mcr <- NULL
alldata_raw_train$zerocross_masd <- NULL
alldata_raw_test$zerocross_masd <- NULL
alldata_raw_train$zerocross_flat <- NULL
alldata_raw_test$zerocross_flat <- NULL

alldata_raw_train$roll.off_mean <- NULL
alldata_raw_test$roll.off_mean <- NULL
alldata_raw_train$roll.off_std <- NULL
alldata_raw_test$roll.off_std <- NULL
alldata_raw_train$roll.off_mcr <- NULL
alldata_raw_test$roll.off_mcr <- NULL
alldata_raw_train$roll.off_masd <- NULL
alldata_raw_test$roll.off_masd <- NULL
alldata_raw_train$roll.off_flat <- NULL
alldata_raw_test$roll.off_flat <- NULL

alldata_raw_train$centroid_mean <- NULL
alldata_raw_test$centroid_mean <- NULL
alldata_raw_train$centroid_std <- NULL
alldata_raw_test$centroid_std <- NULL
alldata_raw_train$centroid_mcr <- NULL
alldata_raw_test$centroid_mcr <- NULL
alldata_raw_train$centroid_masd <- NULL
alldata_raw_test$centroid_masd <- NULL
alldata_raw_train$centroid_flat <- NULL
alldata_raw_test$centroid_flat <- NULL

alldata_raw_train$kurtosis_mean <- NULL
alldata_raw_test$kurtosis_mean <- NULL
alldata_raw_train$kurtosis_std <- NULL
alldata_raw_test$kurtosis_std <- NULL
alldata_raw_train$kurtosis_mcr <- NULL
alldata_raw_test$kurtosis_mcr <- NULL
alldata_raw_train$kurtosis_masd <- NULL
alldata_raw_test$kurtosis_masd <- NULL
alldata_raw_train$kurtosis_flat <- NULL
alldata_raw_test$kurtosis_flat <- NULL

alldata_raw_train$spread_mean <- NULL
alldata_raw_test$spread_mean <- NULL
alldata_raw_train$spread_std <- NULL
alldata_raw_test$spread_std <- NULL
alldata_raw_train$spread_mcr <- NULL
alldata_raw_test$spread_mcr <- NULL
alldata_raw_train$spread_masd <- NULL
alldata_raw_test$spread_masd <- NULL
alldata_raw_train$spread_flat <- NULL
alldata_raw_test$spread_flat <- NULL

alldata_raw_train$flatness_std <- NULL
alldata_raw_test$flatness_std <- NULL
alldata_raw_train$flatness_mean <- NULL
alldata_raw_test$flatness_mean <- NULL
alldata_raw_train$fltaness_mcr <- NULL
alldata_raw_test$fltaness_mcr <- NULL
alldata_raw_train$fltaness_masd <- NULL
alldata_raw_test$fltaness_masd <- NULL
alldata_raw_train$fltaness_flat <- NULL
alldata_raw_test$fltaness_flat <- NULL

alldata_raw_train$skewness_mean <- NULL
alldata_raw_test$skewness_mean <- NULL
alldata_raw_train$skewness_std <- NULL
alldata_raw_test$skewness_std <- NULL
alldata_raw_train$skewness_mcr <- NULL
alldata_raw_test$skewness_mcr <- NULL
alldata_raw_train$skewness_masd <- NULL
alldata_raw_test$skewness_masd <- NULL
alldata_raw_train$skewness_flat <- NULL
alldata_raw_test$skewness_flat <- NULL

#Preprocessing
alldata_unique_train = unique(na.omit(alldata_raw_train)) #delete duplicate and NA entries
alldata_unique_test = unique(na.omit(alldata_raw_test))

class_train = alldata_unique_train[, 66]    #split training data columns into features(cols 1-20) and class(col 21)
data_feats_train = alldata_unique_train[, -66]

class_test = alldata_unique_test[, 66]    #split testing data columns into features(cols 1-20) and class(col 21)
data_feats_test = alldata_unique_test[, -66]

pca_model_train = prcomp(data_feats_train, center = TRUE, scale = TRUE) #PCA, normalized
pca_model_test = prcomp(data_feats_test, center = TRUE, scale = TRUE) #PCA, normalized

summary(pca_model_train)
summary(pca_model_test)

eigenvalues_train = pca_model_train$sdev^2
eigenvectors_train = pca_model_train$rotation
barplot(eigenvalues_train / sum(eigenvalues_train)) #Barplot of components' contribution

data_feats_pc_train <- as.data.frame(predict(pca_model_train, data_feats_train)[, 1:65]) #Keeping feats 1-15
#data_feats_pc_train[, 16:20] <- 0 #Ignore feats 16-20
data_feats_rec_train = data.frame(t(t(as.matrix(data_feats_pc_train) %*%
                                 t(pca_model_train$rotation)) * pca_model_train$scale
                                  + pca_model_train$center)) #apply the transformation
#data_feats_rec_train[, 16:20] <- NULL #Throw Away Components 17-21
data_train = cbind(data_feats_rec_train, class_train) #compiling final training dataset

eigenvalues_test = pca_model_test$sdev^2
eigenvectors_test = pca_model_test$rotation
barplot(eigenvalues_test / sum(eigenvalues_test)) #Barplot of components' contribution

data_feats_pc_test <- as.data.frame(predict(pca_model_test, data_feats_test)[, 1:65]) #Keeping feats 1-15
#data_feats_pc_test[, 16:20] <- 0 #Ignore feats 17-21
data_feats_rec_test = data.frame(t(t(as.matrix(data_feats_pc_test) %*%
                                        t(pca_model_test$rotation)) * pca_model_test$scale
                                    + pca_model_test$center)) #apply the transformation
#data_feats_rec_test[, 16:20] <- NULL #Throw Away Components 17-21
data_test = cbind(data_feats_rec_test, class_test) #compiling final testing dataset

gmm <- MclustDA(data_train[,-66], data_train[,66])

summary.MclustDA(gmm, parameters = FALSE)
summary.MclustDA(gmm, parameters = FALSE, data_test[,-66], data_test[, 66])

pred_train <- predict.MclustDA(gmm)
pred_test <- predict.MclustDA(gmm, data_test[, -66])
opt_acc = Accuracy(pred_test$classification, data_test[,66])
opt_prec = Precision(data_test[, 66], pred_test$classification)
opt_rec = Recall(data_test[, 66], pred_test$classification)
opt_fmeas = 2*(opt_prec*opt_rec)/(opt_prec + opt_rec)

cat("Accuracy: ", opt_acc)
cat("Precision: ", opt_prec)
cat("Recall: ", opt_rec)
cat("F-measure: ", opt_fmeas)

#Designing ROC Curve
roc.curve(data_test[, 66], pred_test$classification)