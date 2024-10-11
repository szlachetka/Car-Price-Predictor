# Import libraries
library(splines)
library(ncvreg)
library(interp)
library(gam)
library(glmnet)
library(tree)
library(randomForest)
library(gbm)

# Import Data set
car_price = read.csv("C:/Users/speed/Desktop/Statistical Learning/R Projects/Car_price_cleaned.csv")
attach(car_price) # attach variables (predictors and response)
dim(car_price) # sample size of 205, 43 variables
names(car_price) # variable names

# Train and Test Set Split
set.seed(2) # save results
n = length(price)
split = 0.7 # 70/30 Split
ii = sample(1:n, floor(split * n))
y_test = car_price[-ii,]$price

# Regularization Setup
x = model.matrix(price ~ . -car_ID -CarName, car_price)[,-1] # remove ID & name
x_train = x[ii,]
y_train = car_price[ii,]$price
grid = 10^seq(10,-2,length = 100) # lambda candidates
cv_lambda_r = cv.glmnet(x_train, y_train, alpha = 0, lambda=grid, data=car_price)
cv_lambda_l = cv.glmnet(x_train, y_train, alpha = 1, lambda=grid, data=car_price)

# Linear Models
linear_model = lm(price ~ horsepower + enginesize + citympg + curbweight, 
                  subset = ((1:n)[ii]), data=car_price)
gam = gam(price ~ s(horsepower,4) + s(enginesize,4) + s(citympg,4) + curbweight
          + horsepower:enginesize:curbweight, data = car_price, 
          subset = ((1:n)[ii]))
ridge_model = glmnet(x_train, y_train, alpha = 0, lambda = cv_lambda_r$lambda.min)
lasso_model = glmnet(x_train, y_train, alpha = 1, lambda = cv_lambda_l$lambda.min)

# Tree Ensemble Methods
B = 500 # average over 500 trees
m = 4 # number of predictors
boost_train = car_price[ii, c("price", "horsepower", "enginesize", 
                              "citympg", "curbweight")] # boosting training set

bagging = randomForest(price ~ horsepower + enginesize + citympg + curbweight, 
                       subset = ((1:n)[ii]), data=car_price, mtry = m, ntree = B)
random_forest = randomForest(price ~ horsepower + enginesize + citympg + 
                             curbweight, subset = ((1:n)[ii]), data=car_price,
                             mtry = sqrt(m), ntree = B)
boosting = gbm(price ~ horsepower + enginesize + citympg + curbweight, 
               data = boost_train, distribution = 'gaussian', n.trees = 5000, 
               interaction.depth = 6, shrinkage = 0.01, cv.folds = 5)
bi = gbm.perf(boosting, method = "cv") # number of iterations chosen by cv

# Summary of Models
summary(linear_model)
summary(gam)
coef(ridge_model)
coef(lasso_model)
bagging_full = randomForest(price ~ horsepower + enginesize + citympg +
                              curbweight, data = car_price, mtry = m, 
                            importance = TRUE) # for variable importance
importance(bagging_full) # output numerical values
varImpPlot(bagging_full) # plot

# Predicted Values
y_hat_lm = predict(linear_model, newdata = car_price[-ii,])
y_hat_gam = predict(gam, newdata = car_price[-ii,])
y_hat_ridge = predict(ridge_model, newx = x[-ii,])
y_hat_lasso = predict(lasso_model, newx = x[-ii,])
y_hat_bagging = predict(bagging, newdata = car_price[-ii,])
y_hat_random_forest = predict(random_forest, newdata = car_price[-ii,])
y_hat_boosting = predict(boosting, newdata = car_price[-ii,], n.trees = bi)

# Test MSE
mean((y_test - y_hat_lm)^2)
mean((y_test - y_hat_gam)^2)
mean((y_test - y_hat_ridge)^2)
mean((y_test - y_hat_lasso)^2)
mean((y_test - y_hat_bagging)^2)
mean((y_test - y_hat_random_forest)^2)
mean((y_test - y_hat_boosting)^2)

# Residuals for Random Forest
print(y_test - y_hat_boosting) # residuals
boxplot(price) # response is right skewed, outliers have a high price


