library(MLmetrics)
library(e1071)
library(scatterplot3d)
library(ROCR)
library(caret)
library(ROSE)
library(mclust)

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

class = alldata_unique[, 21]    #split data columns into features(cols 1-16) and class(col 17)
data_feats = alldata_unique[, -21]
cor(data_feats)

pca_model = prcomp(data_feats, center = TRUE, scale = TRUE) #PCA, normalized

summary(pca_model)

eigenvalues = pca_model$sdev^2
eigenvectors = pca_model$rotation
barplot(eigenvalues / sum(eigenvalues)) #Barplot of components' contribution


data_feats_pc <- as.data.frame(predict(pca_model, data_feats)[, 1:15]) #Keeping feats 1-12
data_feats_pc[, 16:20] <- 0 #Ignore feats 13-16
data_feats_rec = data.frame(t(t(as.matrix(data_feats_pc) %*%
                                 t(pca_model$rotation)) * pca_model$scale
                                  + pca_model$center)) #apply the transformation
data_feats_rec[, 16:20] <- NULL #Throw Away Components 14-16
data = cbind(data_feats_rec, class) #compiling final dataset

#6-fold CV w/ GMM

gmm <- MclustDA(data[,-16], data[,16])
pred <- cvMclustDA(gmm, nfold=6)
str(pred)

opt_accuracy = Accuracy(pred$classification, data[, 16]) #Calculating Optimal Accuracy
opt_precision = Precision(data[, 16], pred$classification, "music") #Calculating Optimal Precision
opt_recall = Recall(data[, 16], pred$classification, "music") #Calculating Optimal Recall
opt_fmeasure = 2*(opt_precision*opt_recall)/(opt_precision + opt_recall) #Calculating F-measure accordingly

cat("Accuracy: ", opt_accuracy)
cat("Precision: ", opt_precision)
cat("Recall: ", opt_recall)
cat("F-measure: ", opt_fmeasure)

#Designing ROC Curve

roc.curve(pred$classification, data[, 16])

