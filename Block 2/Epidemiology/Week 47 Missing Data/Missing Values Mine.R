# number of droped rows 
naprint(na.action(fit))

#The number of missing values per variable in the data is
colSums(is.na(airquality))

fit2 <- lm(Ozone ~ Wind + Solar.R, data = airquality)
naprint(na.action(fit2))
#its 42 

#Changing perspective on missing data
# the standard approach is to delete the missing data

#1.2 Concepts of MCAR, MAR and MNAR
#If the probability of being missing is the same for all cases, then the data
#are said to be missing completely at random (MCAR)

#If the probability of being missing is the same only within groups defined
#by the observed data, then the data are missing at random (MAR).

#MNAR means that the probability of being
#missing varies for reasons that are unknown to us.

#1.3 Ad-hoc solutions
#1.3.1 Listwise deletion   
na.omit() #The procedure eliminates all cases with one or more missing values on the analysis variables
#advantage of complete-case analysis is convenience.
#If the data are not MCAR, listwise deletion can severely bias estimates of
#means, regression coeficients and correlations
#Listwise deletion can introduce inconsistencies in reporting.

#Schafer and Graham (2002, p. 156) cover the middle ground:
#If a missing data problem can be resolved by discarding only a
#small part of the sample, then the method can be quite effective.

#1.3.2 Pairwise deletion
#Pairwise deletion, also known as available-case analysis
#The method calculates the means and (co)variances on all observed data
data <- airquality[, c("Ozone", "Solar.R", "Wind")]
mu <- colMeans(data, na.rm = TRUE)
cv <- cov(data, use = "pairwise")

library(lavaan)
fit <- lavaan("Ozone ~ 1 + Wind + Solar.R
Ozone ~~ Ozone",
              sample.mean = mu, sample.cov = cv,
              sample.nobs = sum(complete.cases(data)))
#The method has also some shortcomings.
#First, the estimates can be biased if the data are not MCAR. Further,
#the covariance and/or correlation matrix may not be positive definite, which
#is requirement for most multivariate procedures.
#Also, pairwise deletion requires numerical data that follow an
# approximate normal distribution

#1.3.3 Mean imputation
#A quick fix for the missing data is to replace them by the mean. 
#library("mice")
imp <- mice(airquality, method = "mean", m = 1,
            maxit = 1) # we can impute mean like this
#Mean imputation is a fast and simple fix for the missing data. However, it
#will underestimate the variance, disturb the relations between variables, bias
#almost any estimate other than the mean and bias the estimate of the mean
#when data are not MCAR.

#1.3.4 Regression imputation
#Regression imputation incorporates knowledge of other variables with the
#idea of producing smarter imputations build a model and impute by that model

data <- airquality[, c("Ozone", "Solar.R")]
imp <- mice(data, method = "norm.predict", seed = 1,
            m = 1, print = FALSE)
xyplot(imp, Ozone ~ Solar.R)

#Regression imputation yields unbiased estimates of the means under MCAR, just
#Moreover, the regression weights are unbiased under MAR if the factors that in
#influence the missingness are part of the regression model.
#correlations are biased upwards variability of the imputed data is systematically underestimated
#Imputations are too good to be true. Regression imputation
#is a recipe for false positive and spurious relations.







