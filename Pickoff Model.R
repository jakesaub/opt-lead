# Import libraries
library(dplyr)
library(ggplot2)
library(plotly)
library(DMwR)
library(mgcv)

# Read CSVs
pickoff <- read.csv("C:\\Users\\jakes\\Desktop\\Jobs\\Angels Baseball Ops\\On the Job\\Lead Distance\\Pickoff Data.csv", header = FALSE)

# Change column headers
names(pickoff) <- c("pickoff","lead_distance","avg_sprint_speed","avg_lead_against","accel","att_per600_r","pitcher_throws","avg_lead")
pickoff <- na.omit(pickoff)

# Formatting error in first cell
pickoff[1,1] = 0

pickoff$pickoff <- factor(pickoff$pickoff)
pickoff$accel <- as.numeric(as.character(pickoff$accel))
pickoff$att_per600_r <- as.numeric(as.character(pickoff$att_per600_r))
pickoff <- na.omit(pickoff)

# Checking structure of dataframes
str(pickoff)

##################### Pickoff model
# Create training and testing sets using k-fold cross-validation (k = 3)
set.seed(1234)
ind_po <- sample(3, nrow(pickoff), replace=TRUE, prob=c(0.33,0.33, 0.33))
pickoff.training1 <- pickoff[ind_po==1 | ind_po==2, 1:8]
pickoff.testing1 <- pickoff[ind_po==3, 1:8]
pickoff.training2 <- pickoff[ind_po==1 | ind_po==3, 1:8]
pickoff.testing2 <- pickoff[ind_po==2, 1:8]
pickoff.training3 <- pickoff[ind_po==2 | ind_po==3, 1:8]
pickoff.testing3 <- pickoff[ind_po==1, 1:8]

pickoff.training1 <- SMOTE(form = pickoff ~ ., data = pickoff.training1, perc.over = 300, perc.under = 5000, k = 10)
pickoff.training2 <- SMOTE(form = pickoff ~ ., data = pickoff.training2, perc.over = 300, perc.under = 5000, k = 10)
pickoff.training3 <- SMOTE(form = pickoff ~ ., data = pickoff.training3, perc.over = 300, perc.under = 5000, k = 10)

# Pickoff freq before and after SMOTE
as.data.frame(table(pickoff.training1$pickoff))
as.data.frame(table(pickoff.adj_training1$pickoff))

# Visualize data
po_plot <- ggplot(pickoff.training1, aes(x = lead_distance, y = pickoff)) +
  geom_point(aes(color = pickoff))
print(po_plot)

# Build and run the GAM model
pickoff.training1$pickoff <- as.numeric(pickoff.training1$pickoff) - 1
pickoff.training2$pickoff <- as.numeric(pickoff.training2$pickoff) - 1
pickoff.training3$pickoff <- as.numeric(pickoff.training3$pickoff) - 1

pickoff.pred1 <- gam(pickoff ~ s(lead_distance, avg_sprint_speed, avg_lead_against), 
                     data = pickoff.training1)
pickoff.pred2 <- gam(pickoff ~ s(lead_distance, avg_sprint_speed, avg_lead_against), 
                     data = pickoff.training2)
pickoff.pred3 <- gam(pickoff ~ s(lead_distance, avg_sprint_speed, avg_lead_against), 
                     data = pickoff.training3)

# Build and run the GLM
pickoff.pred1 <- glm(pickoff ~ lead_distance + avg_sprint_speed + avg_lead_against + accel + att_per600_r + pitcher_throws + avg_lead, 
                     family = binomial, data = pickoff.training1)
pickoff.pred2 <- glm(pickoff ~ lead_distance + avg_sprint_speed + avg_lead_against + accel + att_per600_r + pitcher_throws + avg_lead, 
                     family = binomial, data = pickoff.training2)
pickoff.pred3 <- glm(pickoff ~ lead_distance + avg_sprint_speed + avg_lead_against + accel + att_per600_r + pitcher_throws + avg_lead, 
                     family = binomial, data = pickoff.training3)

pickoff.pred1 <- logistf(pickoff ~ lead_distance + avg_sprint_speed + avg_lead_against + accel + att_per600_r + pitcher_throws, 
                     data = pickoff.training1, firth = TRUE, p1 = TRUE)

pickoff.pred1 <- brglm(pickoff ~ lead_distance + avg_sprint_speed + avg_lead_against + accel + att_per600_r + pitcher_throws, 
                     family = binomial, data = pickoff.training1, p1 = TRUE)
summary(pickoff.pred1)

# Put predictions into test set dataframe
pickoff.testing1$xpo <- predict(pickoff.pred1, newdata = pickoff.testing1, type = "response")
pickoff.testing1$xpo <- ifelse(pickoff.testing1$xpo < 0, 0.00, pickoff.testing1$xpo)

predict(pickoff.pred1, newdata = data.frame(lead_distance = 13, avg_sprint_speed = 25, avg_lead_against = 10, accel = 27, att_per600_r = 11, pitcher_throws = 1), type = "response")


pickoff.testing2$xpo <- predict(pickoff.pred2, newdata = pickoff.testing2, type = "response")
pickoff.testing2$xpo <- ifelse(pickoff.testing2$xpo < 0, 0.00, pickoff.testing2$xpo)

pickoff.testing3$xpo <- predict(pickoff.pred3, newdata = pickoff.testing3, type = "response")
pickoff.testing3$xpo <- ifelse(pickoff.testing3$xpo < 0, 0.00, pickoff.testing3$xpo)

# Combine the three testing sets to complete the total pickoff dataset
pickoff.totalpred <- rbind(pickoff.testing1, pickoff.testing2, pickoff.testing3)
pickoff.totalpred$pickoff <- as.numeric(pickoff.totalpred$pickoff) - 1
pickoff.totalpred <- na.omit(pickoff.totalpred)

## Create Confusion Matrix
table(pickoff.totalpred$pickoff, pickoff.totalpred$xpo > 0.01)

# Pseudo R2
nagelkerke(pickoff.pred1)

# Variable importance
varImp(pickoff.pred1)

# ROC Curve
roc.curve(pickoff.totalpred$pickoff, pickoff.totalpred$xpo, plotit = TRUE)

# Plot the prediction
# Lead Distance and Sprint Speed for xPO
pickoff.predplot <- ggplot(pickoff.totalpred, aes(x = lead_distance, y = avg_sprint_speed))
ggplotly(pickoff.predplot + 
           geom_point(aes(color = xpo)) + 
           scale_colour_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0.5) +
           geom_smooth(method = 'lm'))

# 3D Surface Plot
pickoff_3d <- plot_ly(data = pickoff.totalpred,
                      x = ~lead_distance, 
                      y = ~avg_sprint_speed, 
                      z = ~avg_lead_against) %>% add_markers(color = ~xpo)
pickoff_3d

# Get R-Sq value
pickoff.rsq <- 1 - (sum((pickoff.totalpred$pickoff - pickoff.totalpred$xpo) ^ 2) / sum((pickoff.totalpred$pickoff - mean(pickoff.totalpred$pickoff)) ^ 2))
pickoff.rsq


#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################
#################################### Caught Stealing Model ######################################
# Read CSV
csteal <- read.csv("C:\\Users\\jakes\\Desktop\\Jobs\\Angels Baseball Ops\\On the Job\\Lead Distance\\SB Data.csv", header = FALSE)

# Change column headers
names(csteal) <- c("caught_stealing","lead_distance","avg_sprint_speed","woba","true_sb_pct_r","true_sb_pct_p","true_sb_pct_c","att_per600_c","att_per600_r","accel","throw_accuracy","avg_lead_against","pop_time")
csteal$throw_accuracy <- as.numeric(as.character(csteal$throw_accuracy))
csteal$off_pct <- as.numeric(as.character(csteal$off_pct))
csteal$accel <- as.numeric(as.character(csteal$accel))
csteal <- na.omit(csteal)

# Formatting error in first cell
csteal[1,1] = 0
csteal$caught_stealing <- factor(csteal$caught_stealing)

# Checking structure of dataframe
str(csteal)

# Normalize data
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}
csteal$lead_distance <- normalize(csteal$lead_distance)
csteal$pop_time <- normalize(csteal$pop_time)
csteal$avg_sprint_speed <- normalize(csteal$avg_sprint_speed)
csteal$throw_accuracy <- normalize(csteal$throw_accuracy)
csteal$true_sb_pct <- normalize(csteal$true_sb_pct)
csteal$off_pct <- normalize(csteal$off_pct)


##################### Caught Stealing model
# Create training and testing sets using k-fold cross-validation (k = 3)
set.seed(1234)
ind_cs <- sample(3, nrow(csteal), replace=TRUE, prob=c(0.33,0.33, 0.33))
csteal.training1 <- csteal[ind_cs==1 | ind_cs==2, 1:9]
csteal.testing1 <- csteal[ind_cs==3, 1:9]
csteal.training2 <- csteal[ind_cs==1 | ind_cs==3, 1:9]
csteal.testing2 <- csteal[ind_cs==2, 1:9]
csteal.training3 <- csteal[ind_cs==2 | ind_cs==3, 1:9]
csteal.testing3 <- csteal[ind_cs==1, 1:9]

csteal.training1 <- SMOTE(form = caught_stealing ~ ., data = csteal.training1, perc.over = 500, perc.under = 357, k = 40)
csteal.training2 <- SMOTE(form = caught_stealing ~ ., data = csteal.training2, perc.over = 500, perc.under = 357, k = 40)
csteal.training3 <- SMOTE(form = caught_stealing ~ ., data = csteal.training3, perc.over = 500, perc.under = 357, k = 40)

as.data.frame(table(csteal.training1$caught_stealing))

# Visualize data
cs_plot <- ggplot(csteal.training1, aes(x = lead_distance, y = avg_sprint_speed)) +
  geom_point(aes(color = caught_stealing))
print(cs_plot)

# Build and run the GLM model
csteal.training1$caught_stealing <- as.numeric(csteal.training1$caught_stealing) - 1
csteal.training2$caught_stealing <- as.numeric(csteal.training2$caught_stealing) - 1
csteal.training3$caught_stealing <- as.numeric(csteal.training3$caught_stealing) - 1

csteal.pred1 <- glm(caught_stealing ~ lead_distance + avg_sprint_speed + woba + true_sb_pct_r + true_sb_pct_p + true_sb_pct_c + att_per600_r, 
                    family = binomial, data = csteal.training1)
csteal.pred2 <- glm(caught_stealing ~ lead_distance + avg_sprint_speed + woba + true_sb_pct_r + true_sb_pct_p + true_sb_pct_c + att_per600_r, 
                    family = binomial, data = csteal.training2)
csteal.pred3 <- glm(caught_stealing ~ lead_distance + avg_sprint_speed + woba + true_sb_pct_r + true_sb_pct_p + true_sb_pct_c + att_per600_r,
                    family = binomial, data = csteal.training3)

summary(csteal.pred1)

##################################################################################################
# Build and run the KNN regression model
library(FNN)
library(knitr)

csteal.pred1 <- FNN::knn.reg(train = csteal.training1[,2:9], test = csteal.testing1[,2:9], y = csteal.training1$caught_stealing, k = 50)
csteal.pred2 <- FNN::knn.reg(train = csteal.training2[,2:9], test = csteal.testing2[,2:9], y = csteal.training2$caught_stealing, k = 50)
csteal.pred3 <- FNN::knn.reg(train = csteal.training3[,2:9], test = csteal.testing3[,2:9], y = csteal.training3$caught_stealing, k = 50)

rmse = function(actual, predicted) {
  sqrt(mean((actual - predicted) ^ 2))
}
# define helper function for getting knn.reg predictions
# note: this function is highly specific to this situation and dataset
make_knn_pred = function(k = 1, training, predicting) {
  pred = FNN::knn.reg(train = training[,2:5], 
                      test = predicting[,2:5], 
                      y = as.numeric(as.character(training$caught_stealing)), k = k)$pred
  act  = as.numeric(as.character(predicting$caught_stealing))
  rmse(predicted = pred, actual = act)
}
# define values of k to evaluate
k = c(1, 5, 10, 25, 40, 50, 75, 100, 250)
# get requested train RMSEs
knn_trn_rmse = sapply(k, make_knn_pred, 
                      training = csteal.training1, 
                      predicting = csteal.training1)
# get requested test RMSEs
knn_tst_rmse = sapply(k, make_knn_pred, 
                      training = csteal.training1, 
                      predicting = csteal.testing1)

# determine "best" k
best_k = k[which.min(knn_tst_rmse)]

# find overfitting, underfitting, and "best"" k
fit_status = ifelse(k < best_k, "Over", ifelse(k == best_k, "Best", "Under"))
# summarize results
knn_results = data.frame(
  k,
  round(knn_trn_rmse, 2),
  round(knn_tst_rmse, 2),
  fit_status
)
colnames(knn_results) = c("k", "Train RMSE", "Test RMSE", "Fit?")

# display results
knitr::kable(knn_results, escape = FALSE, booktabs = TRUE)

# Put KNN predictions into test set dataframe
csteal.testing1$xcs <- csteal.pred1$pred
csteal.testing2$xcs <- csteal.pred2$pred
csteal.testing3$xcs <- csteal.pred3$pred

##################################################################################################
# Random Forest
library(randomForest)

csteal.training1$caught_stealing <- as.factor(csteal.training1$caught_stealing)
csteal.training2$caught_stealing <- as.factor(csteal.training2$caught_stealing)
csteal.training3$caught_stealing <- as.factor(csteal.training3$caught_stealing)


csteal.rf1 <- randomForest(caught_stealing ~ ., data=csteal.training1)
csteal.rf1
csteal.rf2 <- randomForest(caught_stealing ~ ., data=csteal.training2, type = 'prob')
csteal.rf3 <- randomForest(caught_stealing ~ ., data=csteal.training3)

# Put predictions into test set dataframe
csteal.testing1$xcs <- predict(csteal.rf1, newdata = csteal.testing1, type = "prob")[,2]


csteal.testing2$xcs <- predict(csteal.rf2, newdata = csteal.testing2, type = "prob")[,2]
#rf2.votes <- predict(csteal.rf2, newdata = csteal.testing2, type = "vote", norm.votes = TRUE)
#csteal.testing2 <- cbind(csteal.testing2, rf2.votes)

csteal.testing3$xcs <- predict(csteal.rf3, newdata = csteal.testing3, type = "prob")[,2]

# Combine the three testing sets to complete the total csteal dataset
csteal.totalpred <- rbind(csteal.testing1, csteal.testing2, csteal.testing3)
csteal.totalpred$caught_stealing <- as.numeric(csteal.totalpred$caught_stealing) - 1
csteal.totalpred <- na.omit(csteal.totalpred)

## Create Confusion Matrix
table(csteal.totalpred$caught_stealing, csteal.totalpred$xcs > 0.5)

# ROC Curve
roc.curve(csteal.totalpred$caught_stealing, csteal.totalpred$xcs, plotit = TRUE)


##################################################################################################

# Put predictions into test set dataframe
csteal.testing1$xcs <- predict(csteal.pred1, newdata = csteal.testing1, type = "response")
csteal.testing1$xcs <- ifelse(csteal.testing1$xcs < 0, 0.00, csteal.testing1$xcs)

predict(csteal.pred1, newdata = data.frame(lead_distance = 10, pop_time = 2, avg_sprint_speed = 25, accel = 28, throw_accuracy = 3, avg_lead_against = 10, woba = .320, true_sb_pct_r = 1, true_sb_pct_p = 1, true_sb_pct_c = 1, att_per600_c = 10, att_per600_r = 10), type = "response")

csteal.testing2$xcs <- predict(csteal.pred2, newdata = csteal.testing2, type = "response")
csteal.testing2$xcs <- ifelse(csteal.testing2$xcs < 0, 0.00, csteal.testing2$xcs)

csteal.testing3$xcs <- predict(csteal.pred3, newdata = csteal.testing3, type = "response")
csteal.testing3$xcs <- ifelse(csteal.testing3$xcs < 0, 0.00, csteal.testing3$xcs)

# Combine the three testing sets to complete the total csteal dataset
csteal.totalpred <- rbind(csteal.testing1, csteal.testing2, csteal.testing3)
csteal.totalpred$caught_stealing <- as.numeric(csteal.totalpred$caught_stealing) - 1
csteal.totalpred <- na.omit(csteal.totalpred)

## Create Confusion Matrix
table(csteal.totalpred$caught_stealing, csteal.totalpred$xcs > 0.5)
table(csteal.testing1$caught_stealing, csteal.pred1$pred > 0.5)
table(csteal.training1$caught_stealing, csteal.pred1$pred > 0.5)

# Plot the prediction
# Lead Distance and Sprint Speed for xCS
csteal.predplot <- ggplot(csteal.totalpred, aes(x = lead_distance, y = avg_sprint_speed))
ggplotly(csteal.predplot + 
           geom_point(aes(color = xcs)) + 
           scale_colour_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0.5) +
           geom_smooth(method = 'lm'))

# 3D Scatterplot for predictions
csteal_3d <- plot_ly(data = csteal.totalpred,
                     x = ~lead_distance, 
                     y = ~avg_sprint_speed, 
                     z = ~pop_time) %>% add_markers(color = ~xcs)
csteal_3d

# 3D Scatterplot for actual
csteal_3d <- plot_ly(data = csteal.totalpred,
                     x = ~lead_distance, 
                     y = ~avg_sprint_speed, 
                     z = ~pop_time) %>% add_markers(color = ~caught_stealing)
csteal_3d

# Get R-Sq value
csteal.rsq <- 1 - (sum((csteal.totalpred$caught_stealing - csteal.totalpred$xcs) ^ 2) / sum((csteal.totalpred$caught_stealing - mean(csteal.totalpred$caught_stealing)) ^ 2))
csteal.rsq

# Pseudo R2
library(rcompanion)
nagelkerke(csteal.pred1)

# Variable importance
library(caret)
varImp(csteal.pred1)

# ROC Curve
library(ROSE)
roc.curve(csteal.totalpred$caught_stealing, csteal.totalpred$xcs, plotit = TRUE)


# Correlation plots
library(corrplot)
cor.pickoff <- cor(pickoff.totalpred[, sapply(pickoff.totalpred, is.numeric)])
corrplot(cor.pickoff, method = 'color')
cor.csteal <- cor(csteal.totalpred[, sapply(csteal.totalpred, is.numeric)])
corrplot(cor.csteal, method = 'color')

#########################################################################################################
#########################################################################################################
#########################################################################################################
#########################################################################################################
##### Create dummy dataframe for all lead distances with league-average values for the other variables
ld <- seq(6, 15, by=.01)
length(ld)
spr_speed <- replicate(901, 25.4904)
pt <- replicate(901, 2.0209)
lead_ag <- replicate(901, 9.8279)
th_acc <- replicate(901, 3.6304)
accel <- replicate(901, 28.389)
woba <- replicate(901, 0.321)
sb_skill_r <- replicate(901, 1)
sb_skill_p <- replicate(901, 1)
sb_skill_c <- replicate(901, 1)
att_per600_r <- replicate(901, 2.6302)
att_per600_c <- replicate(901, 5.0787)
pitcher_throws <- replicate(901, 0.5)

# Pickoff dataframe
po_dummy <- data.frame(ld,spr_speed,lead_ag, accel, att_per600_r, pitcher_throws)
names(po_dummy) <- c("lead_distance", "avg_sprint_speed", "avg_lead_against","accel","att_per600_r","pitcher_throws")
po_dummy$xpo <- predict(pickoff.pred1, newdata = po_dummy, type = "response")

# Caught stealing dataframe
cs_dummy <- data.frame(ld, pt, spr_speed, accel, th_acc, lead_ag, woba, sb_skill_r, sb_skill_p, sb_skill_c, att_per600_c, att_per600_r)
names(cs_dummy) <- c("lead_distance","pop_time","avg_sprint_speed","accel","throw_accuracy","avg_lead_against","woba","true_sb_pct_r","true_sb_pct_p","true_sb_pct_c","att_per600_c","att_per600_r")
cs_dummy$xcs <- predict(csteal.pred1, newdata = cs_dummy, type = "response")

# Comparison dataframe
comb_dummy <- data.frame(ld, po_dummy$xpo, cs_dummy$xcs, po_dummy$xpo + cs_dummy$xcs)
names(comb_dummy) <- c("lead_distance", "xPickoff", "xCaughtStealing", "Sum")
opt_ld <- comb_dummy %>% 
  filter(comb_dummy$Sum == min(comb_dummy$Sum)) %>% 
  select(lead_distance) %>% 
  pull()

# Plot the two graphs overlaid
ggplot(po_dummy, aes(x = lead_distance, y = xpo)) +
  geom_line(color = 'red') +
  geom_line(data = cs_dummy, aes(x = lead_distance, y = xcs), color = 'blue') +
  geom_vline(xintercept = opt_ld, linetype = 'dashed', color = 'green') +
  xlab("Lead Distance") +
  ylab("Out Risk")

################## 
##### Read in average variable values for runners, pitchers, catchers, and batters
# Average Lead Distance by 1B runner
avg_ld <- read.csv("C:\\Users\\jakes\\Desktop\\Jobs\\Angels Baseball Ops\\On the Job\\Lead Distance\\Avg Lead Distance by Runner.csv", header = TRUE)
names(avg_ld) <- c("runner_id","name","avg_lead")
head(avg_ld)
# Average Sprint Speed by runner
avg_ss <- read.csv("C:\\Users\\jakes\\Desktop\\Jobs\\Angels Baseball Ops\\On the Job\\Lead Distance\\Avg Sprint Speed by Runner.csv", header = TRUE)
names(avg_ss) <- c("runner_id","name","avg_sprint_speed")
head(avg_ss)
# Average 10-ft Acceleration by runner
avg_acc <- read.csv("C:\\Users\\jakes\\Desktop\\Jobs\\Angels Baseball Ops\\On the Job\\Lead Distance\\Avg Acceleration by Runner.csv", header = TRUE)
names(avg_acc) <- c("runner_id","name","avg_accel")
head(avg_acc)
# True SB Skill and SB Attempts per 600 PA by runner
avg_sbr <- read.csv("C:\\Users\\jakes\\Desktop\\Jobs\\Angels Baseball Ops\\On the Job\\Lead Distance\\Runner SB Data.csv", header = TRUE)
names(avg_sbr) <- c("runner_id","name","sb","cs","att","plate_appearances","sb_att_per600","sb_pct","true_sb_pct")
avg_sbr <- select(avg_sbr, c("runner_id","name","true_sb_pct","sb_att_per600"))
avg_sbr$true_sb_pct <- avg_sbr$true_sb_pct / 0.724320719
head(avg_sbr)
# Average Pop Time by catcher
avg_pt <- read.csv("C:\\Users\\jakes\\Desktop\\Jobs\\Angels Baseball Ops\\On the Job\\Lead Distance\\Avg Pop Time by Catcher.csv", header = TRUE)
names(avg_pt) <- c("catcher_id","name","avg_pop_time")
head(avg_pt)
# Average Throw Accuracy by catcher
avg_ta <- read.csv("C:\\Users\\jakes\\Desktop\\Jobs\\Angels Baseball Ops\\On the Job\\Lead Distance\\Avg Throwing Accuracy by Catcher.csv", header = TRUE)
names(avg_ta) <- c("catcher_id","name","avg_throw_accuracy")
head(avg_ta)
# True SB SKill and SB Attempts per 600 PA by catcher
avg_sbc <- read.csv("C:\\Users\\jakes\\Desktop\\Jobs\\Angels Baseball Ops\\On the Job\\Lead Distance\\Catcher SB Data.csv", header = TRUE)
names(avg_sbc) <- c("catcher_id","name","sb","cs","batters_faced","sb_att_per600","sb_pct","true_sb_pct")
avg_sbc <- select(avg_sbc, c("catcher_id","name","true_sb_pct","sb_att_per600"))
avg_sbc$true_sb_pct <- avg_sbc$true_sb_pct / 0.724320719
head(avg_sbc)
# Average Lead Against by pitcher
avg_la <- read.csv("C:\\Users\\jakes\\Desktop\\Jobs\\Angels Baseball Ops\\On the Job\\Lead Distance\\Avg Lead Against by Pitcher.csv", header = TRUE)
names(avg_la) <- c("pitcher_id","name","avg_lead_against")
head(avg_la)
# True SB Skill by pitcher
avg_sbp <- read.csv("C:\\Users\\jakes\\Desktop\\Jobs\\Angels Baseball Ops\\On the Job\\Lead Distance\\Pitcher SB Data.csv", header = TRUE)
names(avg_sbp) <- c("pitcher_id","name","sb","cs","batters_faced","sb_att_per600","sb_pct","true_sb_pct")
avg_sbp <- select(avg_sbp, c("pitcher_id","name","true_sb_pct"))
avg_sbp$true_sb_pct <- avg_sbp$true_sb_pct / 0.724320719
head(avg_sbp)
# Pitcher Handedness
hand <- read.csv("C:\\Users\\jakes\\Desktop\\Jobs\\Angels Baseball Ops\\On the Job\\Lead Distance\\Pitcher Handedness.csv", header = TRUE)
names(hand) <- c("pitcher_id","name","throws")
head(hand)
# Average wOBA by batter
avg_woba <- read.csv("C:\\Users\\jakes\\Desktop\\Jobs\\Angels Baseball Ops\\On the Job\\Lead Distance\\Batter wOBA.csv", header = TRUE)
names(avg_woba) <- c("batter_id","name","woba")
head(avg_woba)


runner <- merge(avg_ld, avg_ss, by = c("runner_id","name"))
runner <- merge(runner, avg_acc, by = c("runner_id","name"))
runner <- merge(runner, avg_sbr, by = c("runner_id","name")) %>% arrange(name)

catcher <- merge(avg_pt, avg_ta, by = c("catcher_id","name"))
catcher <- merge(catcher, avg_sbc, by = c("catcher_id","name")) %>% arrange(name)

pitcher <- merge(avg_la, avg_sbp, by = c("pitcher_id","name"))
pitcher <- merge(pitcher, hand, by = c("pitcher_id","name")) %>% arrange(name)

avg_woba <- arrange(avg_woba, name)


find_opt_ld <- function(r, c, p, b){
  #ld <- runner %>% filter(name == r) %>% select(avg_lead_distance)
  ld <- seq(6, 15, by=0.01)
  ald <- unlist(replicate(901, runner %>% filter(name == r) %>% select(avg_lead)), use.names = FALSE)
  ss <- unlist(replicate(901, runner %>% filter(name == r) %>% select(avg_sprint_speed)), use.names = FALSE)
  #return(head(ss))}
  acc <- unlist(replicate(901, runner %>% filter(name == r) %>% select(avg_accel)), use.names = FALSE)
  sb_r <- unlist(replicate(901, runner %>% filter(name == r) %>% select(true_sb_pct)), use.names = FALSE)
  att_r <- unlist(replicate(901, runner %>% filter(name == r) %>% select(sb_att_per600)), use.names = FALSE)
  pt <- unlist(replicate(901, catcher %>% filter(name == c) %>% select(avg_pop_time)), use.names = FALSE)
  ta <- unlist(replicate(901, catcher %>% filter(name == c) %>% select(avg_throw_accuracy)), use.names = FALSE)
  sb_c <- unlist(replicate(901, catcher %>% filter(name == c) %>% select(true_sb_pct)), use.names = FALSE)
  att_c <- unlist(replicate(901, catcher %>% filter(name == c) %>% select(sb_att_per600)), use.names = FALSE)
  la <- unlist(replicate(901, pitcher %>% filter(name == p) %>% select(avg_lead_against)), use.names = FALSE)
  sb_p <- unlist(replicate(901, pitcher %>% filter(name == p) %>% select(true_sb_pct)), use.names = FALSE)
  hnd <- unlist(replicate(901, pitcher %>% filter(name == p) %>% select(throws)), use.names = FALSE)
  woba <- unlist(replicate(901, avg_woba %>% filter(name == b) %>% select(woba)), use.names = FALSE)
  df <- data.frame(ld, ald, ss, acc, sb_r, att_r, pt, ta, sb_c, att_c, la, sb_p, hnd, woba)
  #return(head(df))}
  df$po_pct <- predict(pickoff.pred1, 
                    newdata = data.frame(lead_distance = df$ld,
                                         avg_sprint_speed = df$ss,
                                         avg_lead_against = df$la,
                                         accel = df$acc,
                                         att_per600_r = df$att_r,
                                         pitcher_throws = df$hnd,
                                         avg_lead = df$ald), 
                    type = "response")
  df$cs_pct <- predict(csteal.pred1, 
                    newdata = data.frame(lead_distance = df$ld,
                                         pop_time = df$pt,
                                         avg_sprint_speed = df$ss,
                                         accel = df$acc,
                                         throw_accuracy = df$ta,
                                         avg_lead_against = df$la,
                                         woba = df$woba,
                                         true_sb_pct_r = df$sb_r,
                                         true_sb_pct_p = df$sb_p,
                                         true_sb_pct_c = df$sb_c,
                                         att_per600_c = df$att_c,
                                         att_per600_r = df$att_r), 
                    type = "response")
  df$tot <- df$po_pct + df$cs_pct
  out_risk_plot <- ggplot(df, aes(x = ld, y = po_pct)) +
    geom_line(color = 'red') +
    geom_line(aes(x = ld, y = cs_pct), color = 'blue') +
    geom_vline(xintercept = opt_ld, linetype = 'dashed', color = 'green') +
    xlab("Lead Distance") +
    ylab("Out Risk")
  #return(df)
  ret_list <- list(optimal_lead = df %>% filter(tot == min(tot)) %>% select(ld), 
                   pickoff_pct = df %>% filter(tot == min(tot)) %>% select(po_pct), 
                   caught_stealing_pct = df %>% filter(tot == min(tot)) %>% select(cs_pct),
                   out_risk_plot = out_risk_plot)
  return(ret_list)
}

find_opt_ld("Bour, Justin","Sanchez, Gary","Kershaw, Clayton","Bogaerts, Xander")

predict(csteal.pred1, newdata = data.frame(lead_distance = 10, pop_time = 2, avg_sprint_speed = 25, accel = 28, throw_accuracy = 3, avg_lead_against = 10, woba = .320, true_sb_pct_r = 1, true_sb_pct_p = 1, true_sb_pct_c = 1, att_per600_c = 10, att_per600_r = 10), type = "response")

################################################################################################
################################################################################################
################################################################################################
################################# Create Shiny App #############################################

# Import libraries
library(shiny)
library(tidyr)

ui = fluidPage(
  titlePanel("Optimal Lead Distance"),
  sidebarLayout(
    sidebarPanel(
      selectInput(inputId = "r", label = "Choose 1B runner:", choices = runner$name),
      selectInput(inputId = "p", label = "Choose pitcher:", choices = pitcher$name),
      selectInput(inputId = "c", label = "Choose catcher:", choices = catcher$name),
      selectInput(inputId = "b", label = "Choose batter:", choices = avg_woba$name),
      sliderInput(inputId = "lead_distance", 
                  label = "Lead Distance:",
                  min = 6, max = 15,
                  value = 0.5, step = 0.01)
    ),
    mainPanel(
      plotlyOutput("my_plot") ,
                 #,click = "plot_click"), 
     # verbatimTextOutput("info")
      tableOutput("my_table")
    )
  )
)

server = function(input, output){
  output$my_plot <- renderPlotly({
  r_data <- filter(runner, name == input$r)
  c_data <- filter(catcher, name == input$c)
  p_data <- filter(pitcher, name == input$p)
  b_data <- filter(avg_woba, name == input$b)
  tot_data <- data.frame(
    lead_distance = seq(6, 15, by=0.01),
    avg_lead = unlist(replicate(901,r_data$avg_lead)),
    avg_sprint_speed = unlist(replicate(901,r_data$avg_sprint_speed)),
    accel = unlist(replicate(901,r_data$avg_accel)),
    true_sb_pct_r = unlist(replicate(901,r_data$true_sb_pct)),
    att_per600_r = unlist(replicate(901,r_data$sb_att_per600)),
    pop_time = unlist(replicate(901,c_data$avg_pop_time)),
    throw_accuracy = unlist(replicate(901,c_data$avg_throw_accuracy)),
    true_sb_pct_c = unlist(replicate(901,c_data$true_sb_pct)),
    att_per600_c = unlist(replicate(901,c_data$sb_att_per600)),
    avg_lead_against = unlist(replicate(901,p_data$avg_lead_against)),
    true_sb_pct_p = unlist(replicate(901,p_data$true_sb_pct)),
    pitcher_throws = unlist(replicate(901,p_data$throws)),
    woba = unlist(replicate(901,b_data$woba))
  )
  tot_data$po_pct <- predict(pickoff.pred1, 
                            newdata = tot_data, 
                            type = "response")
  tot_data$cs_pct <- predict(csteal.pred1,
                            newdata = tot_data,
                            type = "response")
  tot_data$tot <- tot_data$po_pct + tot_data$cs_pct
  opt_lead <- tot_data %>% filter(tot == min(tot)) %>% select(lead_distance) %>% unlist()
  opt_po_pct <- tot_data %>% filter(tot == min(tot)) %>% select(po_pct) %>% unlist()
  opt_cs_pct <- tot_data %>% filter(tot == min(tot)) %>% select(cs_pct) %>% unlist()
  
  plot_ly(tot_data, x = ~lead_distance, y = ~po_pct, 
          name = 'Exp Pickoff', 
          type = 'scatter', 
          mode = 'lines',
          line = list(color = 'red', width = 2)) %>%
    add_trace(tot_data, x = ~lead_distance, y = ~cs_pct, 
              name = 'Exp Caught Stealing',
              type = 'scatter',
              mode = 'lines',
              line = list(color = 'blue', width = 2)) %>%
    add_segments(x = ~opt_lead, xend = ~opt_lead,
                 y = 0, yend = 1, 
              name = 'Optimal Lead', 
              line = list(dash = 'dash', color = 'green')) %>%
    add_segments(x = ~input$lead_distance, xend = ~input$lead_distance, 
                 y = 0, yend = 1,
              name = 'Given Lead', 
              line = list(dash = 'dash', color = 'black')) %>%
    layout(title = "Effect of Out Risk by Lead Distance",
           xaxis = list(title = "Lead Distance", range = c(6,15)),
           yaxis = list (title = "Out Risk", range = c(0.0,0.75)))
  })
  
  output$my_table <- renderTable({
    r_data <- filter(runner, name == input$r)
    c_data <- filter(catcher, name == input$c)
    p_data <- filter(pitcher, name == input$p)
    b_data <- filter(avg_woba, name == input$b)
    tot_data <- data.frame(
      lead_distance = seq(6, 15, by=0.01),
      avg_lead = unlist(replicate(901,r_data$avg_lead)),
      avg_sprint_speed = unlist(replicate(901,r_data$avg_sprint_speed)),
      accel = unlist(replicate(901,r_data$avg_accel)),
      true_sb_pct_r = unlist(replicate(901,r_data$true_sb_pct)),
      att_per600_r = unlist(replicate(901,r_data$sb_att_per600)),
      pop_time = unlist(replicate(901,c_data$avg_pop_time)),
      throw_accuracy = unlist(replicate(901,c_data$avg_throw_accuracy)),
      true_sb_pct_c = unlist(replicate(901,c_data$true_sb_pct)),
      att_per600_c = unlist(replicate(901,c_data$sb_att_per600)),
      avg_lead_against = unlist(replicate(901,p_data$avg_lead_against)),
      true_sb_pct_p = unlist(replicate(901,p_data$true_sb_pct)),
      pitcher_throws = unlist(replicate(901,p_data$throws)),
      woba = unlist(replicate(901,b_data$woba))
    )
    tot_data$po_pct <- predict(pickoff.pred1, 
                               newdata = tot_data, 
                               type = "response")
    tot_data$cs_pct <- predict(csteal.pred1,
                               newdata = tot_data,
                               type = "response")
    tot_data$tot <- tot_data$po_pct + tot_data$cs_pct
    opt_lead <- tot_data %>% filter(tot == min(tot)) %>% select(lead_distance) %>% unlist()
    opt_po_pct <- tot_data %>% filter(tot == min(tot)) %>% select(po_pct) %>% unlist()
    opt_cs_pct <- tot_data %>% filter(tot == min(tot)) %>% select(cs_pct) %>% unlist()
    giv_po_pct <- tot_data %>% filter(lead_distance == input$lead_distance) %>% select(po_pct) %>% unlist()
    giv_cs_pct <- tot_data %>% filter(lead_distance == input$lead_distance) %>% select(cs_pct) %>% unlist()
    df1 <- data.frame(lead_distance = opt_lead, 
                      xpickoff = opt_po_pct, 
                      xcaught = opt_cs_pct, 
                      out_risk = opt_po_pct + opt_cs_pct,
                      row.names = "Optimal")
    df2 <- data.frame(lead_distance = input$lead_distance, 
                      xpickoff = giv_po_pct, 
                      xcaught = giv_cs_pct,
                      out_risk = giv_po_pct + giv_cs_pct,
                      row.names = "Given")
    rbind(df1,df2, make.row.names = TRUE)
  },
  rownames = TRUE)
}

ld_shiny <- shinyApp(ui = ui, server = server)
runApp(ld_shiny)