# Import libraries
library(dplyr)
library(ggplot2)
library(plotly)
library(DMwR)
library(mgcv)
library(rcompanion)
library(caret)
library(ROSE)
library(corrplot)
library(shiny)
library(tidyr)

# Set working directory
setwd("C:/Users/jakes/Desktop/Jobs/Angels Baseball Ops/On the Job/Lead Distance")

#################################################################################
#################################################################################
##################### Pickoff model #############################################
#################################################################################
#################################################################################

# Read CSV
pickoff <- read.csv("Pickoff Data.csv", header = FALSE)

# Change column headers
names(pickoff) <- c("pickoff","lead_distance","avg_sprint_speed","avg_lead_against","accel","att_per600_r","pitcher_throws")
pickoff <- na.omit(pickoff)

# Formatting error in first cell
pickoff[1,1] = 0

pickoff$pickoff <- factor(pickoff$pickoff)
pickoff$accel <- as.numeric(as.character(pickoff$accel))
pickoff$att_per600_r <- as.numeric(as.character(pickoff$att_per600_r))
pickoff <- na.omit(pickoff)

# Checking structure of dataframe
str(pickoff)

# Create training and testing sets using k-fold cross-validation (k = 3)
set.seed(1234)
ind_po <- sample(3, nrow(pickoff), replace=TRUE, prob=c(0.33,0.33, 0.33))
pickoff.training1 <- pickoff[ind_po==1 | ind_po==2, 1:7]
pickoff.testing1 <- pickoff[ind_po==3, 1:7]
pickoff.training2 <- pickoff[ind_po==1 | ind_po==3, 1:7]
pickoff.testing2 <- pickoff[ind_po==2, 1:7]
pickoff.training3 <- pickoff[ind_po==2 | ind_po==3, 1:7]
pickoff.testing3 <- pickoff[ind_po==1, 1:7]

# Run SMOTE algorithm
pickoff.training1 <- SMOTE(form = pickoff ~ ., data = pickoff.training1, perc.over = 2000, perc.under = 1190, k = 5)
pickoff.training2 <- SMOTE(form = pickoff ~ ., data = pickoff.training2, perc.over = 2000, perc.under = 1190, k = 5)
pickoff.training3 <- SMOTE(form = pickoff ~ ., data = pickoff.training3, perc.over = 2000, perc.under = 1190, k = 5)

# Pickoff freq after SMOTE
as.data.frame(table(pickoff.training1$pickoff))

# Build and run the GLM model
pickoff.training1$pickoff <- as.numeric(pickoff.training1$pickoff) - 1
pickoff.training2$pickoff <- as.numeric(pickoff.training2$pickoff) - 1
pickoff.training3$pickoff <- as.numeric(pickoff.training3$pickoff) - 1

pickoff.pred1 <- glm(pickoff ~ lead_distance + avg_sprint_speed + avg_lead_against + accel + att_per600_r + pitcher_throws, 
                     family = binomial, data = pickoff.training1)
pickoff.pred2 <- glm(pickoff ~ lead_distance + avg_sprint_speed + avg_lead_against + accel + att_per600_r + pitcher_throws, 
                     family = binomial, data = pickoff.training2)
pickoff.pred3 <- glm(pickoff ~ lead_distance + avg_sprint_speed + avg_lead_against + accel + att_per600_r + pitcher_throws, 
                     family = binomial, data = pickoff.training3)

# Check summary
summary(pickoff.pred1)

# Spot check specific cases
predict(pickoff.pred1, 
        newdata = data.frame(lead_distance = 8, 
                             avg_sprint_speed = 30, 
                             avg_lead_against = 10, 
                             accel = 27, 
                             att_per600_r = 11, 
                             pitcher_throws = 1), 
        type = "response")

# Put predictions into training set dataframe
pickoff.training1$xpo <- predict(pickoff.pred1, newdata = pickoff.training1, type = "response")
pickoff.training1$xpo <- ifelse(pickoff.training1$xpo < 0, 0.00, pickoff.training1$xpo)

pickoff.training2$xpo <- predict(pickoff.pred2, newdata = pickoff.training2, type = "response")
pickoff.training2$xpo <- ifelse(pickoff.training2$xpo < 0, 0.00, pickoff.training2$xpo)

pickoff.training3$xpo <- predict(pickoff.pred3, newdata = pickoff.training3, type = "response")
pickoff.training3$xpo <- ifelse(pickoff.training3$xpo < 0, 0.00, pickoff.training3$xpo)

# Put predictions into test set dataframe
pickoff.testing1$xpo <- predict(pickoff.pred1, newdata = pickoff.testing1, type = "response")
pickoff.testing1$xpo <- ifelse(pickoff.testing1$xpo < 0, 0.00, pickoff.testing1$xpo)

pickoff.testing2$xpo <- predict(pickoff.pred2, newdata = pickoff.testing2, type = "response")
pickoff.testing2$xpo <- ifelse(pickoff.testing2$xpo < 0, 0.00, pickoff.testing2$xpo)

pickoff.testing3$xpo <- predict(pickoff.pred3, newdata = pickoff.testing3, type = "response")
pickoff.testing3$xpo <- ifelse(pickoff.testing3$xpo < 0, 0.00, pickoff.testing3$xpo)

# Combine the three testing sets to complete the total pickoff dataset
pickoff.totalpred <- rbind(pickoff.testing1, pickoff.testing2, pickoff.testing3)
pickoff.totalpred$pickoff <- as.numeric(pickoff.totalpred$pickoff) - 1
pickoff.totalpred <- na.omit(pickoff.totalpred)

########## Model Evaluation
## Visual Confusion Matrix function
draw_confusion_matrix <- function(cm) {
  
  total <- sum(cm$table)
  res <- as.numeric(cm$table)
  
  # set the basic layout
  layout(matrix(c(1,1,2)))
  par(mar=c(2,2,2,2))
  plot(c(100, 345), c(300, 450), type = "n", xlab="", ylab="", xaxt='n', yaxt='n')
  title('CONFUSION MATRIX', cex.main=2)
  
  # create the matrix 
  classes = colnames(cm$table)
  rect(150, 430, 240, 370, col="#74C476")
  text(195, 435, classes[1], cex=1.2)
  rect(250, 430, 340, 370, col="red")
  text(295, 435, classes[2], cex=1.2)
  text(125, 370, 'Predicted', cex=1.3, srt=90, font=2)
  text(245, 450, 'Actual', cex=1.3, font=2)
  rect(150, 305, 240, 365, col="red")
  rect(250, 305, 340, 365, col="#74C476")
  text(140, 400, classes[1], cex=1.2, srt=90)
  text(140, 335, classes[2], cex=1.2, srt=90)
  
  # add in the cm results
  text(195, 400, res[1], cex=1.6, font=2, col='white')
  text(195, 335, res[2], cex=1.6, font=2, col='white')
  text(295, 400, res[3], cex=1.6, font=2, col='white')
  text(295, 335, res[4], cex=1.6, font=2, col='white')
  
  # add in the specifics 
  plot(c(100, 0), c(100, 0), type = "n", xlab="", ylab="", main = "DETAILS", xaxt='n', yaxt='n')
  text(10, 85, names(cm$byClass[1]), cex=1.2, font=2)
  text(10, 70, round(as.numeric(cm$byClass[1]), 3), cex=1.2)
  text(30, 85, names(cm$byClass[2]), cex=1.2, font=2)
  text(30, 70, round(as.numeric(cm$byClass[2]), 3), cex=1.2)
  text(50, 85, names(cm$byClass[5]), cex=1.2, font=2)
  text(50, 70, round(as.numeric(cm$byClass[5]), 3), cex=1.2)
  text(70, 85, names(cm$byClass[6]), cex=1.2, font=2)
  text(70, 70, round(as.numeric(cm$byClass[6]), 3), cex=1.2)
  text(90, 85, names(cm$byClass[7]), cex=1.2, font=2)
  text(90, 70, round(as.numeric(cm$byClass[7]), 3), cex=1.2)
  
  # add in the accuracy information 
  text(30, 35, names(cm$overall[1]), cex=1.5, font=2)
  text(30, 20, round(as.numeric(cm$overall[1]), 3), cex=1.4)
  text(70, 35, names(cm$overall[2]), cex=1.5, font=2)
  text(70, 20, round(as.numeric(cm$overall[2]), 3), cex=1.4)
}

## Create Confusion Matrix
draw_confusion_matrix(confusionMatrix(data = as.factor(ifelse(pickoff.totalpred$xpo > 0.09, 1, 0)),
                reference = as.factor(pickoff.totalpred$pickoff),
                positive = "1"))

draw_confusion_matrix(confusionMatrix(data = as.factor(ifelse(pickoff.training1$xpo > 0.09, 1, 0)),
                                      reference = as.factor(pickoff.training1$pickoff),
                                      positive = "1"))

# Pseudo R2
nagelkerke(pickoff.pred1)

# Variable importance
varImp(pickoff.pred1)

# ROC Curve
roc_po <- roc.curve(pickoff.totalpred$pickoff, pickoff.totalpred$xpo, plotit = TRUE)
roc_po_df <- data.frame(thresholds = roc_po$thresholds, 
                        false_positive = roc_po$false.positive.rate, 
                        true_positive = roc_po$true.positive.rate,
                        diff = roc_po$true.positive.rate - roc_po$false.positive.rate)

# Correlation plot
cor.pickoff <- cor(pickoff.totalpred[, 1:7])
corrplot(cor.pickoff, method = 'color')


#################################################################################
#################################################################################
##################### Caught Stealing model #####################################
#################################################################################
#################################################################################

# Read CSV
csteal <- read.csv("SB Data.csv", header = FALSE)

# Change column headers
names(csteal) <- c("caught_stealing","lead_distance","avg_sprint_speed","woba","true_sb_pct_r","true_sb_pct_p","true_sb_pct_c","att_per600_c","att_per600_r","accel","throw_accuracy","avg_lead_against","pop_time")
csteal$throw_accuracy <- as.numeric(as.character(csteal$throw_accuracy))
csteal$accel <- as.numeric(as.character(csteal$accel))
csteal <- na.omit(csteal)

# Formatting error in first cell
csteal[1,1] = 0
csteal$caught_stealing <- factor(csteal$caught_stealing)

# Create training and testing sets using k-fold cross-validation (k = 3)
set.seed(1234)
ind_cs <- sample(3, nrow(csteal), replace=TRUE, prob=c(0.33,0.33, 0.33))
csteal.training1 <- csteal[ind_cs==1 | ind_cs==2, 1:9]
csteal.testing1 <- csteal[ind_cs==3, 1:9]
csteal.training2 <- csteal[ind_cs==1 | ind_cs==3, 1:9]
csteal.testing2 <- csteal[ind_cs==2, 1:9]
csteal.training3 <- csteal[ind_cs==2 | ind_cs==3, 1:9]
csteal.testing3 <- csteal[ind_cs==1, 1:9]

# Run SMOTE algorithm
csteal.training1 <- SMOTE(form = caught_stealing ~ ., data = csteal.training1, perc.over = 500, perc.under = 320, k = 5)
csteal.training2 <- SMOTE(form = caught_stealing ~ ., data = csteal.training2, perc.over = 500, perc.under = 320, k = 5)
csteal.training3 <- SMOTE(form = caught_stealing ~ ., data = csteal.training3, perc.over = 500, perc.under = 320, k = 5)

# Caught stealing freq after SMOTE 
as.data.frame(table(csteal.training1$caught_stealing))

# Build and run the GLM model
csteal.training1$caught_stealing <- as.numeric(csteal.training1$caught_stealing) - 1
csteal.training2$caught_stealing <- as.numeric(csteal.training2$caught_stealing) - 1
csteal.training3$caught_stealing <- as.numeric(csteal.training3$caught_stealing) - 1

csteal.pred1 <- glm(caught_stealing ~ lead_distance + avg_sprint_speed + woba + true_sb_pct_r + true_sb_pct_p + true_sb_pct_c + att_per600_r + att_per600_c, 
                    family = binomial, data = csteal.training1)
csteal.pred2 <- glm(caught_stealing ~ lead_distance + avg_sprint_speed + woba + true_sb_pct_r + true_sb_pct_p + true_sb_pct_c + att_per600_r + att_per600_c, 
                    family = binomial, data = csteal.training2)
csteal.pred3 <- glm(caught_stealing ~ lead_distance + avg_sprint_speed + woba + true_sb_pct_r + true_sb_pct_p + true_sb_pct_c + att_per600_r + att_per600_c,
                    family = binomial, data = csteal.training3)

# Check summary
summary(csteal.pred1)

# Put predictions into test set dataframe
csteal.testing1$xcs <- predict(csteal.pred1, newdata = csteal.testing1, type = "response")
csteal.testing1$xcs <- ifelse(csteal.testing1$xcs < 0, 0.00, csteal.testing1$xcs)

csteal.testing2$xcs <- predict(csteal.pred2, newdata = csteal.testing2, type = "response")
csteal.testing2$xcs <- ifelse(csteal.testing2$xcs < 0, 0.00, csteal.testing2$xcs)

csteal.testing3$xcs <- predict(csteal.pred3, newdata = csteal.testing3, type = "response")
csteal.testing3$xcs <- ifelse(csteal.testing3$xcs < 0, 0.00, csteal.testing3$xcs)

# Combine the three testing sets to complete the total csteal dataset
csteal.totalpred <- rbind(csteal.testing1, csteal.testing2, csteal.testing3)
csteal.totalpred$caught_stealing <- as.numeric(csteal.totalpred$caught_stealing) - 1
csteal.totalpred <- na.omit(csteal.totalpred)

###### Model evaluation
## Create Confusion Matrix
draw_confusion_matrix(confusionMatrix(data = as.factor(ifelse(csteal.totalpred$xcs > 0.28, 1, 0)),
                                      reference = as.factor(csteal.totalpred$caught_stealing),
                                      positive = "1")) 

# Pseudo R2
nagelkerke(csteal.pred1)

# Variable importance
varImp(csteal.pred1)

# ROC Curve
roc.curve(csteal.totalpred$caught_stealing, csteal.totalpred$xcs, plotit = TRUE)

# Correlation plot
cor.csteal <- cor(csteal.totalpred[, 1:9])
corrplot(cor.csteal, method = 'color')


#################################################################################
#################################################################################
########################## Quantify results #####################################
#################################################################################
#################################################################################

##### Read in average variable values for runners, pitchers, catchers, and batters
# Average Sprint Speed by runner
avg_ss <- read.csv("Avg Sprint Speed by Runner.csv", header = TRUE)
names(avg_ss) <- c("runner_id","name","avg_sprint_speed")
head(avg_ss)
# Average 10-ft Acceleration by runner
avg_acc <- read.csv("Avg Acceleration by Runner.csv", header = TRUE)
names(avg_acc) <- c("runner_id","name","avg_accel")
head(avg_acc)
# True SB Skill and SB Attempts per 600 PA by runner
avg_sbr <- read.csv("Runner SB Data.csv", header = TRUE)
names(avg_sbr) <- c("runner_id","name","sb","cs","att","plate_appearances","sb_att_per600","sb_pct","true_sb_pct")
avg_sbr <- select(avg_sbr, c("runner_id","name","true_sb_pct","sb_att_per600"))
avg_sbr$true_sb_pct <- avg_sbr$true_sb_pct / 0.724320719
head(avg_sbr)
# Average Pop Time by catcher
avg_pt <- read.csv("Avg Pop Time by Catcher.csv", header = TRUE)
names(avg_pt) <- c("catcher_id","name","avg_pop_time")
head(avg_pt)
# Average Throw Accuracy by catcher
avg_ta <- read.csv("Avg Throwing Accuracy by Catcher.csv", header = TRUE)
names(avg_ta) <- c("catcher_id","name","avg_throw_accuracy")
head(avg_ta)
# True SB SKill and SB Attempts per 600 PA by catcher
avg_sbc <- read.csv("Catcher SB Data.csv", header = TRUE)
names(avg_sbc) <- c("catcher_id","name","sb","cs","batters_faced","sb_att_per600","sb_pct","true_sb_pct")
avg_sbc <- select(avg_sbc, c("catcher_id","name","true_sb_pct","sb_att_per600"))
avg_sbc$true_sb_pct <- avg_sbc$true_sb_pct / 0.724320719
head(avg_sbc)
# Average Lead Against by pitcher
avg_la <- read.csv("Avg Lead Against by Pitcher.csv", header = TRUE)
names(avg_la) <- c("pitcher_id","name","avg_lead_against")
head(avg_la)
# True SB Skill by pitcher
avg_sbp <- read.csv("Pitcher SB Data.csv", header = TRUE)
names(avg_sbp) <- c("pitcher_id","name","sb","cs","batters_faced","sb_att_per600","sb_pct","true_sb_pct")
avg_sbp <- select(avg_sbp, c("pitcher_id","name","true_sb_pct"))
avg_sbp$true_sb_pct <- avg_sbp$true_sb_pct / 0.724320719
head(avg_sbp)
# Pitcher Handedness
hand <- read.csv("Pitcher Handedness.csv", header = TRUE)
names(hand) <- c("pitcher_id","name","throws")
head(hand)
# Average wOBA by batter
avg_woba <- read.csv("Batter wOBA.csv", header = TRUE)
names(avg_woba) <- c("batter_id","name","woba")
head(avg_woba)

################# Create dataframes of aforementioned values per player for each of four actors
runner <- merge(avg_ss, avg_acc, by = "runner_id")
runner <- merge(runner, avg_sbr, by = "runner_id") %>% arrange(name) %>% select(runner_id,name,avg_sprint_speed, avg_accel, true_sb_pct, sb_att_per600)

catcher <- avg_sbc %>% arrange(name)

pitcher <- merge(avg_la, avg_sbp, by = "pitcher_id")
pitcher <- merge(pitcher, hand, by = "pitcher_id") %>% arrange(name) %>% select(pitcher_id,name,avg_lead_against, true_sb_pct, throws)

avg_woba <- arrange(avg_woba, name)

################# Write function to retrieve optimal lead distance given the runner, catcher, pitcher, and batter
find_opt_ld <- function(r, c, p, b){
  lead_distance <- seq(6, 15, by=0.01)
  avg_sprint_speed <- unlist(rep(runner %>% filter(name == r) %>% select(avg_sprint_speed), 901), use.names = FALSE)
  accel <- unlist(rep(runner %>% filter(name == r) %>% select(avg_accel), 901), use.names = FALSE)
  true_sb_pct_r <- unlist(rep(runner %>% filter(name == r) %>% select(true_sb_pct), 901), use.names = FALSE)
  att_per600_r <- unlist(rep(runner %>% filter(name == r) %>% select(sb_att_per600), 901), use.names = FALSE)
  true_sb_pct_c <- unlist(rep(catcher %>% filter(name == c) %>% select(true_sb_pct), 901), use.names = FALSE)
  att_per600_c <- unlist(rep(catcher %>% filter(name == c) %>% select(sb_att_per600), 901), use.names = FALSE)
  avg_lead_against <- unlist(rep(pitcher %>% filter(name == p) %>% select(avg_lead_against), 901), use.names = FALSE)
  true_sb_pct_p <- unlist(rep(pitcher %>% filter(name == p) %>% select(true_sb_pct), 901), use.names = FALSE)
  pitcher_throws <- unlist(rep(pitcher %>% filter(name == p) %>% select(throws), 901), use.names = FALSE)
  woba <- unlist(rep(avg_woba %>% filter(name == b) %>% select(woba), 901), use.names = FALSE)
  df <- data.frame(lead_distance, avg_sprint_speed, accel, true_sb_pct_r, 
                   att_per600_r, true_sb_pct_c, att_per600_c, avg_lead_against, 
                   true_sb_pct_p, pitcher_throws, woba)
  df$po_pct <- predict(pickoff.pred1, 
                       newdata = df, 
                       type = "response")
  df$cs_pct <- predict(csteal.pred1, 
                       newdata = df, 
                       type = "response")
  df$tot <- df$po_pct + df$cs_pct
  optimal_lead <- unlist(df %>% filter(tot == min(tot)) %>% select(lead_distance))
  ret_list <- list(optimal_lead, min(df$tot))
  return(ret_list)
}

########## Read in data of all 2B stolen base attempts since 2015 with actual lead distances
ald_data <- read.csv("Individual Lead Distances on 2B SB Attempts.csv", header = TRUE)
names(ald_data) <- c("runner","catcher","pitcher","batter","lead_distance","caught_stealing")

### Find the optimal lead distances in each situation found by the function
ald_data$opt_lead <- apply(ald_data[,c("runner","catcher","pitcher","batter")], 
                           1, 
                           function(x) find_opt_ld(x["runner"], x["catcher"], x["pitcher"], x["batter"])[[1]])

ald_data$out_risk <- apply(ald_data[,c("runner","catcher","pitcher","batter")], 
                           1, 
                           function(x) find_opt_ld(x["runner"], x["catcher"], x["pitcher"], x["batter"])[[2]])

ggplot(ald_data, aes(x = lead_distance, y = opt_lead)) +
  geom_point(aes(color = out_risk)) +
  scale_colour_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0.5) +
  xlim(6,15) +
  ylim(6,15)

# Boxplot of distrbutions of actual lead distance vs. optimal
ggplot(ald_data, aes(x = "Actual", y = lead_distance)) +
  geom_boxplot() + 
  geom_boxplot(data = ald_data, aes(x = "Optimal", y = opt_lead)) + 
  scale_y_continuous(breaks=seq(0,30,2)) + 
  xlab("Distribution") + 
  ylab("Lead Distance") +
  ggtitle("Lead Distance Distribution Comparison") +
  theme(plot.title = element_text(hjust = 0.5))

rbind(summary(ald_data$lead_distance), summary(ald_data$opt_lead))

### Compare player optimal lead distances to actual lead distances
lead_diff <- ald_data %>% 
  group_by(runner) %>% 
  summarise(attempts = n(),
            avg_act_lead = mean(lead_distance), 
            avg_opt_lead = mean(opt_lead),
            diff = avg_opt_lead - avg_act_lead,
            exp_out_risk = mean(out_risk))

cs <- ald_data %>% 
  group_by(runner) %>% 
  filter(caught_stealing == 1) %>%
  count()

lead_diff <- left_join(lead_diff, cs, by = "runner")
names(lead_diff)[7] <- "cs"

sb <- ald_data %>% 
  group_by(runner) %>% 
  filter(caught_stealing == 0) %>%
  count()

lead_diff <- left_join(lead_diff, sb, by = "runner")
names(lead_diff)[8] <- "sb"

lead_diff$cs_pct <- lead_diff$cs / (lead_diff$attempts)
lead_diff <- select(lead_diff,-c(cs, sb))
lead_diff <- filter(lead_diff, attempts >= 10)


ggplot(lead_diff, aes(x = avg_act_lead, y = avg_opt_lead)) +
  geom_point(aes(color = exp_out_risk)) +
  scale_colour_gradient2(low = "blue", mid = "white", high = "red")

runner_lead <- merge(lead_diff, runner, by.x = "runner", by.y = "name")

# Average sprint speed by optimal lead distance
ggplot(runner_lead, aes(x = avg_sprint_speed, y = avg_opt_lead)) + 
  geom_point(aes(color = exp_out_risk)) + 
  geom_smooth(method = "lm") +
  xlab("Avg Sprint Speed") + 
  ylab("Avg Optimal Lead") + 
  ggtitle("Optimal Lead by Sprint Speed") + 
  theme(plot.title = element_text(hjust = 0.5))

# Average sprint speed by actual lead distance
ggplot(runner_lead, aes(x = avg_sprint_speed, y = avg_act_lead)) + 
  geom_point(aes(color = exp_out_risk)) + 
  geom_smooth(method = "lm") +
  xlab("Avg Sprint Speed") + 
  ylab("Avg Actual Lead") + 
  ggtitle("Actual Lead by Sprint Speed") + 
  theme(plot.title = element_text(hjust = 0.5))

# Correlations
cor(runner_lead$avg_sprint_speed, runner_lead$avg_opt_lead)
cor(runner_lead$avg_sprint_speed, runner_lead$avg_act_lead)


#################################################################################
#################################################################################
########################## Create Shiny app #####################################
#################################################################################
#################################################################################

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
                  value = 10, step = 0.01)
    ),
    mainPanel(
      plotlyOutput("my_plot") ,
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
      avg_sprint_speed = unlist(rep(r_data$avg_sprint_speed,901)),
      accel = unlist(rep(r_data$avg_accel)),
      true_sb_pct_r = unlist(rep(r_data$true_sb_pct,901)),
      att_per600_r = unlist(rep(r_data$sb_att_per600,901)),
      true_sb_pct_c = unlist(rep(c_data$true_sb_pct,901)),
      att_per600_c = unlist(rep(c_data$sb_att_per600,901)),
      avg_lead_against = unlist(rep(p_data$avg_lead_against,901)),
      true_sb_pct_p = unlist(rep(p_data$true_sb_pct,901)),
      pitcher_throws = unlist(rep(p_data$throws,901)),
      woba = unlist(rep(b_data$woba,901))
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
      avg_sprint_speed = unlist(rep(r_data$avg_sprint_speed,901)),
      accel = unlist(rep(r_data$avg_accel)),
      true_sb_pct_r = unlist(rep(r_data$true_sb_pct,901)),
      att_per600_r = unlist(rep(r_data$sb_att_per600,901)),
      true_sb_pct_c = unlist(rep(c_data$true_sb_pct,901)),
      att_per600_c = unlist(rep(c_data$sb_att_per600,901)),
      avg_lead_against = unlist(rep(p_data$avg_lead_against,901)),
      true_sb_pct_p = unlist(rep(p_data$true_sb_pct,901)),
      pitcher_throws = unlist(rep(p_data$throws,901)),
      woba = unlist(rep(b_data$woba,901))
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

# Run Shiny app
ld_shiny <- shinyApp(ui = ui, server = server)
runApp(ld_shiny)
