?fdgs
head(fdgs)
sum(is.na(fdgs))

#The number of missing values per variable in the data is

colSums(is.na(fdgs))
#Since the goal is to estimate the mean weight of the 
#population, and we assume MAR, we have to impute 
#(complete case analysis is only unbiased under MCAR). 
#Regression imputation will work fine for this purpose. 
#For the imputation model, I would create a linear model 
#with wgt as the outcome. The model will include all variables 
#(except id, and hgt.z) and also a quadratic effect of age 
#(as weight will probably increase less as the child gets older).
#Then, I can impute the predicted values for the missing weight 
#cells. The sample mean will then be the estimate of the 
#population mean
df <- fdgs
# create the imputation model with a quadratic term for age
imp_model <- lm(wgt ~ reg + age + I(age^2) + sex + hgt, data = df, na.action = na.omit)

# it's good to look at the R^2 to see how good the prediction is!
summary(imp_model)$adj.r.squared
# the R-squared is very high: 0.9298. That gives us quite good certainty around the imputed values. So let's do the imputation!
na_idx <- is.na(df$wgt)
pred_wgt <- predict(imp_model, newdata = df[na_idx,])
pred_wgt
# we can impute the predicted values like so:
df$wgt[na_idx] <- pred_wgt

# mean estimate with regression imputation:
mean(df$wgt)
# mean estimate with complete case analysis:
mean(df$wgt[-na_idx])

#Extra challenge: now, using the same model, perform 
#stochastic regression imputation (norm.nob) as explained 
#in section 1.3.5 of FIMD and compute the sample mean of 
#weight. Do it again. Is the result the same? What does 
#this variation in the sample mean represent?

# We have to add noise to the prediction. We sample the value 
#we pick from a normal distribution, where the mean of the distribution 
#is the prediction, and the variance is the residual variance from the model.

# this is how to get the residual standard deviation (sqrt(variance))
res_sd <- summary(imp_model)$sigma
# first, predict the means as in normal regression imputation
pred_wgt <- predict(imp_model, newdata = df[na_idx,])

# then, add noise by sampling from a normal distribution with mean 0 and sd sigma
pred_wgt <- pred_wgt + rnorm(length(pred_wgt), sd = res_sd)

# then we impute and compute the mean!
df$wgt[na_idx] <- pred_wgt
mean(df$wgt)
# From the slides: what we see is between-dataset variance: the extra variance 
#caused by the fact that there are missing values in the sample


