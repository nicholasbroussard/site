---
author: Nick Broussard
date: "2019-09-28"
title: Short Term Rentals in Austin
subtitle: A Case Study using Binomial Logistic Regression
---

In Austin, short-term rental (STR) hosts who comply with regulations are far more likely to be penalized than hosts that do not comply. Specifically, delaying payment by just one month decreases the likelihood you’ll ever have pay by 20%. Additionally, larger fees are less likely to be paid. Hosts are 50% less likely to pay their fine for every $50 increase in the total fine amount. 

Hosts choose not to comply for several reasons. First, STR regulations are confusing. Just understanding the STR category a host falls under requires scouring municipal code. As a reference, there are three STR types. Type 1 is owner-occupied,  Type 2 is not owner-occupied and not part of a multifamily unit, and Type 3 is not owner-occupied but is part of a multifamily unit.  Understanding basic STR categories is
Additionally, fee amounts and explanations are not available to the public – at least not in any reasonable way.  A resident might get a mailed fee request, but there’s no way to validate the

Second, STR hosting is expensive. A first-time STR license costs \$572 and annual renewals cost \$313.  Licensing costs are the same regardless of the STR type. All STR types pay the same licensure fees. 

Third, and most importantly, the City does not enforce compliance. They dont have the money, and therefore don't have the manpower, to send inspectors all over the city to verify permits. 

Because of these factors, it’s is likely that, in 2019, Austin failed to collect $2.5M in fees. According to the Austin Code Department, “over 10,000 properties in Austin advertise as STRs, but only about 2,500 of them are licensed.”  This is especially pertinent given the recent decision by Texas’ Appeals Court regarding Austin’s ban on short term rentals for “non-homestead residential properties by 2022”  as unconstitutional.

The City of Austin hosts a great data portal. I found a dataset (released from the municipal court?) that details judicial decisions related to STR violations. You can find scripts in R and Python [on my Github](https://github.com/nicholasbroussard/tutorials/tree/master/austin_str_binomial_logistic), or just follow along with the R code below. Some of the code is abbreviated, so if you really want to replicate my data, make sure you take a peek at Github.

````r
library(tidyverse)

df <- read.csv("https://raw.githubusercontent.com/nicholasbroussard/tutorials/master/austin_str_binomial_logistic/austin_str_binomial_logistic.csv",
            stringsAsFactors = F)
```

You can find the original dataset [here](https://data.austintexas.gov/City-Government/Short-Term-Rental-Legal-Outcomes/xfmy-m22z)

I calculated the length of time for each case to run its course... that is, the time from a case being opened to finally being closed.  

```r 
df <- df %>% 
  mutate(MONTHSVIOLATIONOPEN = interval(VIOLATIONOPENDATE, LEGALOUTCOMEDATE)%/%months(1)) %>%
  select(-VIOLATIONOPENDATE, -LEGALOUTCOMEDATE)
```

There's a big spread, so I figured it has something to do with the type of case. There are generally four types of violations that lead to a case: 

* Occupancy Limit Violations
* Length of Stay Violations
* Incomplete Registration Violations
* Licensing Violations

```r 
df$DEFICIENCYTEXT <- as.character(df$DEFICIENCYTEXT)
df <- df %>%
  mutate(DEFICIENCYTEXT = case_when(str_detect(DEFICIENCYTEXT, "must obtain a license")
  str_detect(DEFICIENCYTEXT, "is not licensed") ~ "Licensing Violation",
  str_detect(DEFICIENCYTEXT, "may not advertise or promote") ~ "Incomplete Registration Violation",
  str_detect(DEFICIENCYTEXT, "may not be used by more than") | str_detect(DEFICIENCYTEXT, "may not include the rental of less than") | str_detect(DEFICIENCYTEXT, "16 people") ~ "Occupancy Limit Violation",
  str_detect(DEFICIENCYTEXT, "The advertisement required a three night minimum stay") ~ "Length of Stay Violation"))
```
Ultimately, I want to know if a case actually resulted in a penalty. I can then dig into the weeds regarding why certain cases are penalized while others aren't.

```r
df <- df %>%
  mutate(LEGALOUTCOME = ifelse(LEGALOUTCOME=="Liable" | LEGALOUTCOME=="Closed due to Judicial / Admin Action", 0, 1))
#0 is "not penalized", 1 is penalized. I have to code the outcome var as 0/1 for the glm().
df <- df %>%
  select(LEGALOUTCOME, DEFICIENCYTEXT, OUTSTANDINGFEE, MONTHSVIOLATIONOPEN)
```
After some exploratory data analysis and cleaning (using the package [inspectdf](https://github.com/alastairrushworth/inspectdf)... so awesome!) I can move into my predictive modeling phase. 

First, I split my data: 60% to train the data, and the remainder to test data. The R package [caTools](https://rdrr.io/cran/caTools/man/sample.split.html) works like a charm.  

```r 
library(caTools)
set.seed(101) #The seed number does not matter. Any number will do.
sample <- sample.split(df, SplitRatio = .6)
test <- subset(df, sample == TRUE)
train <- subset(df, sample == FALSE)
```
After splitting, I can run my first model. I'm going to run a binary logistic regression, with the case's legal outcome as my dependent variable. 

For the outcome variable, 0 is not penalized and 1 is penalized. Also, the reference category for DEFICIENCYTEXT is RegistrationViolation.

```r 
set.seed(101)
model <- glm(LEGALOUTCOME ~ ., data = train, family = binomial(link="logit")) 
summ(model, exp = T, vifs = T, digits = 4)
```
![](model1_output.jpg)

We're on the right track! All my variables are significant except for Licensing Violation. Based on the exponentiated coefficients (which gives odds instead of log odds), we find...

* An occupancy limit violation is 11x (1,098%) more likely to result in a penalty than an incomplete registration violation.
* For every $1 increase in outstanding fee the likelihood of penalty decreases by 1% (1-.989).
* For every month that the violation is open the likelihood of penalty decreases by 19% (1-.81).

![](coefficients.png)

There's a lot of different ways to assess the model's goodness of fit, including AIC, residual analysis, and the ROC curve, but I find a basic confusion matrix to be the easiest to understand. 

![](model1_confusion_matrix.jpg)

So from the looks of things, our model is a pretty solid fit. (In terms of logistic regression, our model minimizes the deviation between the outcome variables and the predictor)

No we can ask the model questions... questions like "If my case has been opened for six months, is it likely that I'll ever have to pay anything?" Answer: No. Or, "If I owe $900, is it likely that I'll get to walk away without paying a dime?" Answer: Yes.

And why is this important? Wel,, if we take a step back, we can see that the City of Austin has lost a considerable amount of money because of it's inability to prosecute expediently. Cases that are open for more than five months and fines in the 50th percentile have a low likelihoodof ever being paid. 

![](oddsgraph_feegraph.png)

Bottom line: If you're running Airbnb out of your home, just trying to make a little extra cash, and you get fined by the City - odds are that you'll ultimately have to pay the fine. But if you're a big conglomerate running lots of Airbnb's illegally out of a condominum complex and you have the legal support to seek judicial delays... you'll probably never pay a penny.