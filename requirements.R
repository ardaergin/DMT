# Packages
library(dplyr)
library(tidyr)
library(ggplot2)
library(patchwork)
library(scales)
library(lubridate)
library(Matrix)
# Time Series
library(zoo)
# Imputation for Time Series
library(imputeTS)
# Network Analysis
library(mlVAR)
library(igraph)

# Bayesian Multilevel
library(mHMMbayes)

# Multiple Imputations
library(mice)
library(micemd)
library(miceadds)

# Modelling
library(lme4)
library(performance)



### FUNCTIONS ###
fix_time <- function(data){
  data_to_write <- data
  data_to_write <- data_to_write %>% mutate(
    date_time = format(date_time, format="%Y-%m-%d %H:%M:%S"))
  data_to_write$date_time <- as.character(data_to_write$date_time)
  return(data_to_write)
}

# Function to calculate mean only if not all values are NA
safe_mean <- function(x) {
  if (all(is.na(x))) NA else mean(x, na.rm = TRUE)
}

# Function to calculate sum only if not all values are NA
safe_sum <- function(x) {
  if (all(is.na(x))) NA else sum(x, na.rm = TRUE)
}
