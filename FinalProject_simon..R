#Final Project by Simon Estepa.


library(boot)
library(tidyverse)
library(caret)
library(ROCR)
library(dplyr)
library(stargazer)
library(pROC)

#PART A

# Dataset
df <- read.delim("C:/Users/simon/Downloads/hmda_sw.txt")

# Selecting the variables based on financial intuition and renaming them.
df <- df %>%
  rename(
    Loan_Type = s3,
    Loan_Purpose = s4,
    Occupancy = s5,
    Loan_Amount = s6,
    Loan_Approval = s7,
    County = s11,
    Applicant_Race = s13,
    Applicant_Sex = s15,
    Applicant_Income = s17,
    Loan_Purchaser_Type = s18,
    Denial_Reason_Original = s19a,
    Denial_Reason_Additions = s19b,
    Denial_Reason_Corrections = s19c,
    Denial_Reason_Other = s19d,
    Units_Property = s20,
    Credit_History_MP = s42,
    Debt_to_Income_Ratio_Total_Obligations = s46,
    Term_of_Loan_Months = s48,
    Appraised_Value = s50,
    Gift_or_Grant_for_Down_Payment = s54,
    Co_signer_for_Application = s55,
    Owner_Occupied_Property = dnotown,
    Type_of_Property = dprop,
    Education = school,
    Marital_Status = s23a,
    Self_Employed = s27a
  )
head(df)



# PART B 

# Display summary statistics for the selected variables in the data frame
summary_stats <- summary(df[c("Loan_Amount", "Debt_to_Income_Ratio_Total_Obligations", "Credit_History_MP")])
print(summary_stats)

# Mean calculations
mean_loan_amount <- mean(df$Loan_Amount, na.rm = TRUE)
mean_dti <- mean(df$Debt_to_Income_Ratio_Total_Obligations, na.rm = TRUE)
mean_credit_history <- mean(df$Credit_History_MP, na.rm = TRUE)

# Average representative applicant values
cat("\nMean Loan Amount: ", mean_loan_amount)
cat("\nMean Debt-to-Income Ratio: ", mean_dti)
cat("\nMean Credit History: ", mean_credit_history)

#Credit History mortgage payments codes:
#1- no late mortgage payments
#2- No mortgage payments history
#3- one or two late mortgage payments
#4- more than two late mortgage payments

# histograms of interest.
par(mar = c(5, 4, 4, 2) + 0.1)

# Histogram for Loan Amount
hist(df$Loan_Amount, main="Loan Amount", xlab="Loan Amount", col="yellow", border="black")
# the average loan amount is $138k per client

# Histogram for Debt to Income Ratio 
hist(df$Debt_to_Income_Ratio_Total_Obligations, main="Debt to Income Ratio ", xlab="Debt to Income Ratio", col="green", border="black")
# there is an average of 33% of debt to income ratio which is good.

# Histogram for Credit History (Mortgage Payments)
hist(df$Credit_History_MP, main="Credit History (Mortgage Payments)", xlab="Credit History", col="red", border="black")
# Based on the data description around 750 have no late mortgage payments
# around 1600 have no mortgage payment history
# very few have 1 or 2 mortgage payments and around 1% have 2 or more late mortgage payments.

# PART C

#filtering to keep levels 1 and 3 in Loan_Approval
df <- df[df$Loan_Approval %in% c(1, 3), ]

# binary 'approve' variable.
# whether the values in the "Loan_Approval" column are either 1 or 2. 
#If yes, it assigns 1; otherwise, it assigns 0.
df$approve <- ifelse(df$Loan_Approval == 1 | df$Loan_Approval == 2,1,0)
attach(df)

# Logistic Regression Model
logistic_model_one <- glm(approve ~ Debt_to_Income_Ratio_Total_Obligations + Applicant_Race + Self_Employed + 
                               Education + Loan_Amount, data = df, family = "binomial")

# Model summary
summary(logistic_model_one)

# ROC Curve and AUC
roc_curve <- roc(df$approve, predict(logistic_model_one, type = "response"))
auc_value <- auc(roc_curve)
print (auc_value)

# Plot ROC Curve
plot(roc_curve, main = "ROC Curve", col = "blue", lwd = 2)
abline(a = 0, b = 1, lty = 2, col = "gray")  

# AUC value on the plot
text(0.8, 0.2, paste("AUC =", round(auc_value, 3)), col = "blue", cex = 0.8)

# Confusion Matrix at Alternative Cut-off Levels 
cutoff_levels <- seq(0.1, 0.9, by = 0.1)
for (cutoff in cutoff_levels) {
  predicted_labels <- ifelse(predict(logistic_model_one, type = "response") > cutoff, 1, 0)
  confusion_matrix <- table(Actual = df$approve, Predicted = predicted_labels)
  
  # Metrics Calculation
  sensitivity <- confusion_matrix[2, 2] / sum(confusion_matrix[2, ])
  specificity <- confusion_matrix[1, 1] / sum(confusion_matrix[1, ])
  false_positive_rate <- confusion_matrix[1, 2] / sum(confusion_matrix[1, ])
  false_negative_rate <- confusion_matrix[2, 1] / sum(confusion_matrix[2, ])
  accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
  error_rate <- 1 - accuracy
  
  # Print Metrics for Each Cut-off Level
  cat("\nCut-off Level:", cutoff, "\n")
  cat("Sensitivity:", sensitivity, "\n")
  cat("Specificity:", specificity, "\n")
  cat("False Positive Rate:", false_positive_rate, "\n")
  cat("False Negative Rate:", false_negative_rate, "\n")
  cat("Model Accuracy:", accuracy, "\n")
  cat("Model Error Rate:", error_rate, "\n")
}

#SUMMARY OF FINDINGS
#In summary, The analysis focuses on levels 1 and 3 in the Loan_Approval variable.
#the model suggests that factors like debt-to-income ratio, applicant race, 
#and self-employment status significantly influence loan approval. However, education and 
#loan amount have limited impact .
#Significance codes in model summary.
#*** (0.001): Highly significant
#** (0.01): Significant
#* (0.05): Marginally significant
#*
#. (0.1): Not significant


#PART D

#. 4 models where estimated with their threshold value included.

costfunc = function(approve, pred_prob){
  weight1 = 1               # The weight for approve=1 but classified as 0 (false negative)
  weight0 = 1               # The weight for approve=0 but classified as 1 (false positive)
  c1 <- (approve==1)&(pred_prob<optimal_cutoff)     # The 'count' for approve=1 but classified as 0 (false negatives)
  c0 <- (approve==0)&(pred_prob>=optimal_cutoff)    # The 'count' for approve=0 but classified as 1 (false positives)
  cost <- mean(weight1*c1 + weight0*c0)
  return(cost)              
} 

#set seed.
set.seed(123)

# This models explores the relationship between loan approval (approve) and various predictors,
# considering interactions and polynomial terms to capture potential non-linearities.

#basic linear variables no interactions or polynomials included.
model_1 <- glm(approve ~ Debt_to_Income_Ratio_Total_Obligations * Credit_History_MP +
                        Loan_Amount * Applicant_Income +
                        Loan_Amount +  
                        Units_Property + Loan_Purpose +
                        Loan_Type + Owner_Occupied_Property,
                      data = df, family = binomial)


model_2 <- glm(approve ~ Debt_to_Income_Ratio_Total_Obligations + Credit_History_MP + Applicant_Sex +
                 poly(Applicant_Income, 2, raw = TRUE) + 
                 Units_Property + Loan_Type,
               data = df, family = binomial)

#considered both linear and non linear relationships. including transformed variables like Self_Employed, Appraised_Value,
#Applicant_Income, Credit_History_MP, and Debt_to_Income_Ratio_Total_Obligations. 
#Interaction terms, such as Loan_Amount * poly(Self_Employed), are included to capture joint effects.
#We used quadratic terms to allow a more flexible representation of its loan approval relationship
#Even tough the mr was similar to other models. statistics metrics increased as we used quadratic terms
#big portion of the approach that we took was based on summary statistics of different variables.
model_3 <- glm(approve ~ Loan_Amount + poly(Self_Employed, 2, raw = TRUE) + poly(Appraised_Value, 2, raw = TRUE) + 
                 Loan_Amount * poly(Self_Employed, 2, raw = TRUE) + poly(Applicant_Income, 2, raw = TRUE) +
                 poly(Credit_History_MP, 2, raw = TRUE) + poly(Debt_to_Income_Ratio_Total_Obligations, 2, raw = TRUE),
               data = df, family = binomial)


model_4<- glm(approve ~ Debt_to_Income_Ratio_Total_Obligations + 
                     poly(as.numeric(Credit_History_MP), 2, raw=TRUE) +
                     poly(as.numeric(Self_Employed), 2, raw=TRUE) + 
                     Loan_Amount + county + Loan_Purchaser_Type + Type_of_Property,
                   data=df, family=binomial)


threshold_seq <- seq(0.01, 1, 0.01) 
misclassification_rate = rep(0, length(threshold_seq))  

for(i in 1:length(threshold_seq)){ 
  optimal_cutoff = threshold_seq[i]
  set.seed(123)
  misclassification_rate[i] = cv.glm(data=df, glmfit=model_3, cost = costfunc, K=10)$delta[1]
}

plot(threshold_seq, misclassification_rate)

optimal_cutoff_cv = threshold_seq[which(misclassification_rate==min(misclassification_rate))]
optimal_cutoff_cv
min(misclassification_rate)



#TABLE DOCUMENTING THE PERFORMANCE OF VARIOUS MODELS.


# Define the sequence of threshold values
threshold_seq <- seq(0.01, 1, 0.01) 

# Initialize a matrix to store misclassification rates for each model and threshold
misclassification_rates <- matrix(NA, nrow = length(threshold_seq), ncol = 4)

# Loop through each model
for (j in 1:4) {
  # Selected the appropriate model based on the loop index
  model <- switch(j,
                  model_1,
                  model_2,
                  model_3,
                  model_4
  )
  
  # Loop through each threshold value
  for (i in 1:length(threshold_seq)) { 
    # Set the current threshold value
    optimal_cutoff = threshold_seq[i]
    
    # Performed 10-fold cross-validation and stored the misclassification rate
    set.seed(123)
    misclassification_rates[i, j] = cv.glm(data=df, glmfit=model, cost = costfunc, K=10)$delta[1]
  } 
}

# Assigned column names to the misclassification rates matrix
colnames(misclassification_rates) <- c("Model 1", "Model 2", "Model 3", "Model 4")

# Calculated the minimum misclassification rates for each model
min_misclassification_rates <- apply(misclassification_rates, 2, min)

# Identified the indices of optimal cutoffs for each model
optimal_cutoff_indices <- sapply(apply(misclassification_rates, 2, which.min), function(x) ifelse(all(is.na(x)), NA, x[1]))

# Extracted the optimal cutoff values from the threshold sequence
optimal_cutoffs <- ifelse(is.na(optimal_cutoff_indices), NA, threshold_seq[optimal_cutoff_indices])

#We created a results table with model names, minimum misclassification rates, and optimal cutoffs
result_table <- data.frame(
  Model = c("Model 1", "Model 2", "Model 3", "Model 4"),
  Min_Misclassification_Rate = min_misclassification_rates,
  Optimal_Cutoff = optimal_cutoffs
)

# result table
result_table

  