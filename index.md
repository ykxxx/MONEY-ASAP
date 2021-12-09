## Welcome to Your Salary Estimator!

Wanna lead a perfect life without money worries? 🤑🤑🤑

---

As a group of aspiring data scientists, we are highly interested and motivated to embark on our career in tech companies. Before seeking a full-time job, we want to learn more about the salaries from different companies. While some of us have started on our internships, the rest would like to also navigate the data science and STEM career options as possible alternatives. 

## Purpose

We would like to explore more about income and realise our own place predict from analysis of the data. Specifically, we will be analyzing a set of factors that are potentially predictive of the outcome of us earning $100k/yr upon graduation. We will be using a selection of methods to visualise the data as well as reduce the dimension before applying logistic gression, K-nearest neighbor, decision tree, random forest to make predictions. The data would also be presented in a shiny app. Apart from our own money concerns, we would also like to learn about the representativeness of certain socioeconomic groups affect their salary negotiation, and whether our project could help with uncovering the transparency lacking in this dynamic.

Our preferred features include: company, level, title, education level and some other personal backgrounds.

## Data

We used [Kaggle Data Science and STEM income dataset](https://www.kaggle.com/jackogozaly/data-science-and-stem-salaries) for our analysis. This is a dataset scraped from the website of levels.fyi, a site with a visualised, crowdsourced database for transparent leveling charts across companies to help with job negotiation and possibly prevent underpaying for those underrepresented within tech industry.

The original dataset contains 62642 records. We filtered the redundant features and kept `company`, `title`, `totalyearlycompensation`, `location`, `yearsofexperience`, `yearsatcompany`, `basesalary`, `stockgrantvalue`, `bonus`, `gender`, `cityid`, `dmaid`, `Race`, `Education`, and corrected datatype.

We wanted to see the base salary of different titles.

<img src="https://github.com/Nancy-dvZhang/MONEY-ASAP/raw/main/images/basesalary_title.png" width="95%"/>  

Does race matter?

<img src="https://github.com/Nancy-dvZhang/MONEY-ASAP/raw/main/images/basesalary_race.png" width="95%"/>  

What about education?

<img src="https://github.com/Nancy-dvZhang/MONEY-ASAP/raw/main/images/basesalary_edu.png" width="95%"/>  

## Final Analysis
<p>
Shown below is a brief overview of the analysis conducted by team A$AP Money (<i>click to expand</i>):
</p>
<details>
<summary><b>Logistic Regression</b></summary>
<br>
<p>
Company and location information is excluded from our multinomial logistic regression model because they are categorical data with so many categories that they obscure our results. First, we fit a full model with all covariates we selected to see the general relationship pattern. We excluded the individuals with NA entries in either Education, Race, and gender to tidy up the data and to make trends more visible. Another reason we decide to drop these entries is that without them, we still have enough data (>20,000) to draw strong prediction power. 
</p>
<p>
<pre class="r"><code class="hljs">df_1 &lt;- filtered_df %&gt;% select(c(<span class="hljs-string">"title"</span>, <span class="hljs-string">"yearsofexperience"</span>, <span class="hljs-string">"yearsatcompany"</span>,<span class="hljs-string">'gender'</span>,<span class="hljs-string">'cityid'</span>,<span class="hljs-string">'dmaid'</span>,<span class="hljs-string">'Race'</span>,<span class="hljs-string">'Education'</span>, <span class="hljs-string">'y'</span>)) %&gt;% filter(!is.na(Education) &amp; !is.na(gender) &amp; !is.na(Race))
mod_logistic_whole = glm(y ~ ., data = df_1, family=binomial())
summary(mod_logistic_whole)</code></pre>
</p>
<p>
From our full model, we see that all of our covariates provide helpful information with as least one category being significant except gender. We then proceed to bidirectional step-wise selection procedure to further feature selection and applied complete case analysis (CCA) where there is data missingness in the category missing completely at random (MCAR). Upon completion of feature selection and running the model again with the final set of covariates, we generated a confusion matrix as follows:
</p>
```
<p>
<pre class="r"><code>set.seed(1)
x &lt;- stratified(df_2, &quot;y&quot;, 0.8, keep.rownames = TRUE)
train_set &lt;- x %&gt;% dplyr::select(-rn)
train_index &lt;- as.numeric(x$rn)
test_set &lt;- df_2[-train_index,]
</code></pre>
</p>
<p>
<pre class="r"><code>mod_logistic_train = glm(y ~ ., data = train_set, family=binomial())
preds_hat &lt;- predict(mod_logistic_train, newdata = test_set, type= &quot;response&quot;)
preds &lt;- ifelse(preds_hat&gt;0.5,1,0)
preds &lt;- as.numeric(substring(preds,-1))
confusionMatrix(data = as.factor(preds), reference = as.factor(test_set$y), positive = &quot;1&quot;)</code></pre>
</p>
<pre><code>## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    0    1
##          0  715  201
##          1  393 3027
##                                           
##                Accuracy : 0.863           
##                  95% CI : (0.8524, 0.8731)
##     No Information Rate : 0.7445          
##     P-Value [Acc &gt; NIR] : &lt; 2.2e-16       
##                                           
##                   Kappa : 0.6182          
##                                           
##  Mcnemar&#39;s Test P-Value : 4.621e-15       
##                                           
##             Sensitivity : 0.9377          
##             Specificity : 0.6453          
##          Pos Pred Value : 0.8851          
##          Neg Pred Value : 0.7806          
##              Prevalence : 0.7445          
##          Detection Rate : 0.6981          
##    Detection Prevalence : 0.7887          
##       Balanced Accuracy : 0.7915          
##                                           
##        &#39;Positive&#39; Class : 1               
## </code></pre>
<p>
Our logistic regression model yields an overall accuracy of 0.863 with 95% CI (0.8524, 0.8731) and a very small p-value, which indicates that we did an OK job in predicting basesalary >= 100K. However, our sensitivity is 0.9377 and our specificity is 0.6453, giving a balanced accuracy of 0.7915. This means our logistic regression prediction model is making more Type 1 error (False positive) than type 2 errors.
</p>
</details>

<details>
<summary><b>Data Manipulation for KNN, Decision Tree, and Random Forest</b></summary>
<br>
<p>
In this section, we closely examined the dataset and applied dimension reduction (clustering), refined classification, data cleaning, further EDA of distribution, factorization where appropriate.
</p>

</details>
  

<details>
<summary><b>K-Nearest Neighbors (KNN)</b></summary>
<br>
  
<pre class="r"><code>set.seed(42)
x &lt;- stratified(dataset, &quot;y&quot;, 0.8, keep.rownames = TRUE)
train_set &lt;- x %&gt;% dplyr::select(-rn)
train_index &lt;- as.numeric(x$rn)
test_set &lt;- dataset[-train_index,]
print(dim(train_set))</code></pre>
<pre><code>## [1] 17273    12</code></pre>
<pre class="r"><code>print(dim(test_set))</code></pre>
<pre><code>## [1] 4318   12</code></pre>
<pre class="r"><code>set.seed(42)
knn_fit &lt;- train_set %&gt;% knn3(y~., data = ., k = 12)
knn_hat &lt;- predict(knn_fit, newdata = test_set, type = &quot;class&quot;)
knn_p &lt;- f_hat2 &lt;- predict(knn_fit, newdata = test_set, k=12)[,2]
confusionMatrix(data = knn_hat, reference = as.factor(test_set$y), positive = &quot;1&quot;)</code></pre>
<pre><code>## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    0    1
##          0  827  161
##          1  276 3054
##                                           
##                Accuracy : 0.8988          
##                  95% CI : (0.8894, 0.9076)
##     No Information Rate : 0.7446          
##     P-Value [Acc &gt; NIR] : &lt; 2.2e-16       
##                                           
##                   Kappa : 0.7245          
##                                           
##  Mcnemar&#39;s Test P-Value : 4.943e-08       
##                                           
##             Sensitivity : 0.9499          
##             Specificity : 0.7498          
##          Pos Pred Value : 0.9171          
##          Neg Pred Value : 0.8370          
##              Prevalence : 0.7446          
##          Detection Rate : 0.7073          
##    Detection Prevalence : 0.7712          
##       Balanced Accuracy : 0.8498          
##                                           
##        &#39;Positive&#39; Class : 1               
## </code></pre>
                   
</details>
  

<details>
<summary><b>Decision Tree</b></summary>
<br>
  
<p>
We then apply several decision trees to visualize the prediction of the majority class of our outcome (whether one receives >$100k/yr base salary) based within the decision process partitioning based on a combination of variables. We selected a set of covariates that could potentially be predictive of the outcome variable. After combinatory attempts to construct reasonable decision processes, three combinations resulted in clear partition of majority class in the outcome variable:
</p>
<img src="https://github.com/Nancy-dvZhang/MONEY-ASAP/raw/main/images/4_dec_tree_1.png" width="95%"/>  
<p>
In the first tree, the outcome is partitioned by years of experience < 11, city id as a proxy for the importance of locations, and years of experience < 5.
</p>
<img src="https://github.com/Nancy-dvZhang/MONEY-ASAP/raw/main/images/4_dec_tree_2.png" width="95%"/>  
<p>
In the second decision tree, if years of experience > 5, we predict the outcome to be >$100k/yr base salary. If not, if employee number of the company is larger than 506, the outcome category is also predicted to be >\$100k/yr base salary. The next divisions are made by years of experience > 3 and emplyee number > 136.
</p>
<img src="https://github.com/Nancy-dvZhang/MONEY-ASAP/raw/main/images/4_dec_tree_3.png" width="95%"/>  
<p>
The third tree predicts the outcome by the same cutoffs in years of experience. The rest of the division lies in size category and title.
</p>                       
</details>
  
<details>
<summary><b>Random Forest</b></summary>
<br>
<p> 
We build a random forest model with 3 features for each tree (~ sqrt level)
</p>
<pre class="r"><code>set.seed(42)
rf_fit &lt;- randomForest(y ~ .-state-country, data = train_set, mtry = 3)
rf_hat &lt;- predict(rf_fit, newdata = test_set, type = &quot;class&quot;)
rf_p &lt;- predict(rf_fit, newdata = test_set, type = &quot;prob&quot;)[,2]
confusionMatrix(data = rf_hat, reference = as.factor(test_set$y), positive = &quot;1&quot;)</code></pre>
<pre><code>## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    0    1
##          0  870  162
##          1  233 3053
##                                          
##                Accuracy : 0.9085         
##                  95% CI : (0.8995, 0.917)
##     No Information Rate : 0.7446         
##     P-Value [Acc &gt; NIR] : &lt; 2.2e-16      
##                                          
##                   Kappa : 0.7543         
##                                          
##  Mcnemar&#39;s Test P-Value : 0.0004282      
##                                          
##             Sensitivity : 0.9496         
##             Specificity : 0.7888         
##          Pos Pred Value : 0.9291         
##          Neg Pred Value : 0.8430         
##              Prevalence : 0.7446         
##          Detection Rate : 0.7070         
##    Detection Prevalence : 0.7610         
##       Balanced Accuracy : 0.8692         
##                                          
##        &#39;Positive&#39; Class : 1              
## </code></pre>
<p>Now, check the feature importance</p>
<pre class="r"><code>rf_im &lt;- importance(rf_fit)
as.matrix(rf_im[order(-rf_im[,1]),])</code></pre>
<pre><code>##                         [,1]
## dmaid             2013.06633
## cityid            1501.71226
## yearsofexperience  853.07254
## title              539.38443
## size_category      398.53391
## yearsatcompany     331.52472
## Race               280.51223
## Education          209.93064
## gender              84.32126</code></pre>
</details>
<details>
<summary><b>tSNE Visualization</b></summary>
<br>
<p>
Here, we do a visualization using tSNE. First, to reduce computational burden, we randomly sample 5000 points for visualization. At the same time, since `cityid` and `dmaid` are two strongest factors, we only keep `dmaid` to avoid overfitting. let's see the pattern of different labels:
</p>
<img src="https://github.com/Nancy-dvZhang/MONEY-ASAP/raw/main/images/6_tSNE_1.png" width="95%"/>  
<p>
Let's see the pattern of the strongest factor `dmaid`
</p>               
<img src="https://github.com/Nancy-dvZhang/MONEY-ASAP/raw/main/images/6_tSNE_2.png" width="95%"/>  

</details>

<details>
<summary><b>Linear Regression</b></summary>
<br>
<H4>Explore Dataset</H4>
<p>  
We also wanted to fit the Dataset into a linear regression model so that we will be able to make predictions on base salary given information about a person's relevant features. So in this analysis, we explored the performance of different feature engineering and encoding methods, as well as the prediction performance of several linear regression model. We will choose the best model to make our base salary predictor in the next section. We further examined the dataset for the purpose of the linear model, and some of our considerations include:
</p>
<p>
Since the location in this datafram is encoded as City, State, Country (except for locations in the US, where it is encoded as City, State), we would like to split this into 3 separate features and take a look into the distribution of countries. 
</p>
<p>
After ploting the base salary for different countries, we can see that there is a large discrapencies between each countries.
</p>
<img src="https://github.com/Nancy-dvZhang/MONEY-ASAP/raw/main/images/7_country_discrp.png" width="95%"/>  
<p>
If we list the number of data we have for each countries, we can also see that the dataset is very imbalanced -- the majoriety of data comes from the US, where as other countries only contribute a very small amount of data, this can be a major problem for machine learning models, since there may not be enough data for the model to learn the representation for other countries properly. Therefore, we decided to filter only data from the US, and train the linear regression model as a base salary predictor for the US.
</p>
<img src="https://github.com/Nancy-dvZhang/MONEY-ASAP/raw/main/images/7_us_data_display.png" width="95%"/>  
<H4>Feature Engineering</H4>
<p>
Next we would like to proceed to the feature engineering stage. Since we are building the linear regression model to use it as the base salary predictor for our Shiny App, we can only include features that we want users to input for their salary prediction. This means that we cannot use feature such as `dmaid` in the model since we don't know what this feature represent. So after looking at our EDA plots, we decided to choose 6 features: `title`, `company`, `yearsatcompany`, `yearsofexperience`, `Education`, and `Race` as our features. 
</p>
<p>
Considering the ease of calculation for our Shiny App Predictor, we also decided to use a categorical encoding for each feature instead of one-hot encoding, even though empirically one-hot encoding may yield better results. 
</p>
<p>
To do the feature engineering for `title`, we plot the average base salary for each title again. We can see that there is a clear difference in average base salary for different title. To make sure that our title encoding can capture this pattern of differences, we decided to first encode each title category as their average base salary, and later we would also normalize this encoding value to range between [0, 1], and this process was repeated for the other 5 features.
</p>
<p>
For the feature `company`, we know that this dataset includes data from 1633 distinct conpaies, so we think it would probably be better it we convert `company` into new new feature with less levels. To do this, we first take a look into how the `company` feature distribution looks like. After visual examination and summaries of the data, we see that the number of appearance a company has in this dataset, `n`, actually highly correlates with the actual company size. For instance, the major tech companies like Amazon, Micorsoft, Google, Facebook, and Apple ranks top 5 in the tabke above, and all has over 1000 data entries. So we decided to use this `n` create a new feature called `company_size` and use this feature for our model training. After the above was completed for our selected features, we joined the data and exported the cleaned dataframe into a .csv file and also save all the encoding data so that we can use them for reference when we later calculate the predicted base salary in our Shiny App Predictor.
</p>
<H4>Training the Linear Regression model</H4>
<p>
Fitting the training dataset into the linear regression model, we saw see that all the features are significant, this suggests that they are all important and contribute to the final prediction of base salary. So we now save these coefficient for our base salary predictor.
</p>
<pre class="r"><code>lm1 &lt;- lm(basesalary ~ ., data = train)
summary(lm1)</code></pre>
<pre><code>## 
## Call:
## lm(formula = basesalary ~ ., data = train)
## 
## Residuals:
##     Min      1Q  Median      3Q     Max 
## -136537  -22387   -2887   17936  706237 
## 
## Coefficients:
##                Estimate Std. Error t value Pr(&gt;|t|)    
## (Intercept)       69438       1756  39.543  &lt; 2e-16 ***
## company           31005       1332  23.281  &lt; 2e-16 ***
## title             46921       2199  21.340  &lt; 2e-16 ***
## education         21241       1340  15.847  &lt; 2e-16 ***
## race               8973       1501   5.976 2.34e-09 ***
## experience       125164       2032  61.598  &lt; 2e-16 ***
## yearsatcompany   -30720       3576  -8.589  &lt; 2e-16 ***
## ---
## Signif. codes:  0 &#39;***&#39; 0.001 &#39;**&#39; 0.01 &#39;*&#39; 0.05 &#39;.&#39; 0.1 &#39; &#39; 1
## 
## Residual standard error: 39170 on 13562 degrees of freedom
## Multiple R-squared:  0.3474, Adjusted R-squared:  0.3471 
## F-statistic:  1203 on 6 and 13562 DF,  p-value: &lt; 2.2e-16</code></pre>
<p>To evaluate model performance, we choose to calculate its RMSE.</p>
<pre class="r"><code>RMSE &lt;- function(true_ratings, predicted_ratings){
    sqrt(mean((true_ratings - predicted_ratings)^2))
}</code></pre>
<p>The RMSE for the test dataset prediction is relatively large, however, considering the noise that naturally exists in this dataset, where people with the exact same features is likely to have different base salary, a large testing RMSE doesn’t necessarily suggest that our model is performing bad, and we would need further analysis to fairly evaluate our model performance.</p>
<pre class="r"><code>lm_pred &lt;- predict(lm1, newdata = test)
lm_rmse &lt;- RMSE(lm_pred, test$basesalary)
lm_rmse</code></pre>
<pre><code>## [1] 44100.67</code></pre>
<p>By comparing our model prediction and the actual base salary in the testing dataset, we can see that while not accurate, our model is able to capture the general trend of base salary level. This means that the prediction of out model can at least provide some reference for a person’s potential base salary based on their input.</p>
<pre class="r"><code>lm_pred[1:10]</code></pre>
<pre><code>##        1       11       12       17       23       25       26       33 
## 162817.7 123619.4 151725.9 169328.8 190601.6 163993.0 173833.8 205034.7 
##       35       37 
## 153926.6 134367.7</code></pre>
<pre class="r"><code>test$basesalary[1:10]</code></pre>
<pre><code>##  [1] 210000 112000 114000 153000 215000 160000 210000 150000 180000 165000</code></pre>
                   
</details>
  

<details>
<summary><b>Optimizing Model Performance - One-hot Encoding and Regularization</b></summary>
<br>
<p>
To better evaluate our previous model performance, we also want to explore whether we can optimize the previous simple Linear Regression model through one-hot encoding and regularization. To do this, we prepared the one-hot encoded dataset, splited training and testing dataset, fitted a linear regression model using the one-hot encoded dataset, and evaluted model performance using RMSE on testing dataset. The RMSE is slightly lower than our previous linear regression model, which suggests that one-hot encoding do yield better results. However, since the difference is pretty small, we would still use the weight from our previous model for the ease of calculation. To explore the effect of regularization, we also fit the one-hot encoded data into a ridge regression model. However, the RMSE is higher for ridge regression, so we would not choose this model. 
</p>
<p>
Overall, our first simple linear regression exhibits a reasonably good prediction on base salary, so we decided to use its weight to build our final salary predictor shiny app.
</p>

</details>
  

## Our Discovery

Method|Accuracy|Sensitivity|Specificity
---|---|---|---
Logistic Regression|0.863| 0.9377|0.6453
KNN| 0.8988|**0.9499**|0.7498
Random Forest|**0.9085**|0.9496|**0.7888**

All models can make a sensible prediction with a relatively higher sensitivity then specificity. It shows that our models work better predicting `TRUE` labels, which represent getting a job with salary higher than 100k. `Random Forest` achieves the best performance, because it can better deal with overfitting and the problem with imbalanced data.

## Let's predict!

We built a [shiny app](https://ykxxx.shinyapps.io/predictor/) for users to predict their salaries.

<details>
<summary><b>Shiny APP Implementation</b> (<i>Click to Expand</i>)</summary>
<br>
<p>We implemented 3 main features in our Shiny App, which are embedded in 3 seperate tabs.</p>
<div id="estimator" class="section level4">
<h4>Estimator</h4>
<p>User can input their information for each of the 5 features in the drop-down bar, and our salary estimator will return a table of salary estimation as well as the confidence intercal for each of the 15 titles.</p>
</div>
<div id="boxplot" class="section level4">
<h4>Boxplot</h4>
<p>User can view the boxplot of base salary distribution for each of the 6 features</p>
</div>
<div id="salary-distribution-plot" class="section level4">
<h4>Salary distribution plot</h4>
<p>User can select a base salary upper bound, and view the distribution of salary.</p>
<pre class="r"><code>library(shiny)
library(tidyverse)
library(ggplot2)

# library(rsconnect)
# rsconnect::deployApp(&#39;/Users/kexinyang/Desktop/BST 260/Project/notebook.Rmd&#39;)

df &lt;- read.csv(&quot;project/clean_us_salary_data.csv&quot;)
coef &lt;- read.csv(&quot;project/lm_coef.csv&quot;)
race_encode &lt;- read.csv(&quot;project/race_encode.csv&quot;)
company_encode &lt;- read.csv(&quot;project/company_encode.csv&quot;)
education_encode &lt;- read.csv(&quot;project/education_encode.csv&quot;)
title_encode &lt;- read.csv(&quot;project/title_encode.csv&quot;)
experience_encode &lt;- read.csv(&quot;project/experience_encode.csv&quot;)
yearsatcompany_encode &lt;- read.csv(&quot;project/yearsatcompany_encode.csv&quot;)

ui &lt;- fluidPage(
    
    theme = shinythemes::shinytheme(&quot;flatly&quot;),

    titlePanel(&quot;Salary Estimator :)&quot;),
    
    tabsetPanel(
        tabPanel(&quot;Estimator&quot;,
                 sidebarLayout(
                     sidebarPanel(&quot;Choose your features:&quot;, 
                        fluidRow(selectInput(&quot;race&quot;, label = &quot;Select a race&quot;, selected = &quot;Asian&quot;,
                                           choices = race_encode$x)),
                        fluidRow(selectInput(&quot;company&quot;, label = &quot;Select a company&quot;, selected = &quot;Amazon&quot;,
                                           choices = company_encode$x)),
                        fluidRow(selectInput(&quot;education&quot;, label = &quot;Select a education&quot;,
                                           choices = education_encode$x)),
                        fluidRow(selectInput(&quot;yearsofexperience&quot;, label = &quot;Select a years of experience&quot;, selected = &quot;1&quot;,
                                           choices = experience_encode$x)),
                        fluidRow(selectInput(&quot;yearsatcompany&quot;, label = &quot;Select a years at company&quot;, selected = &quot;0&quot;,
                                           choices = yearsatcompany_encode$x))
                     ),
                     mainPanel(
                         tableOutput(&quot;estimated_salary&quot;)
                     )
                 ),
                 ),
        tabPanel(&quot;BoxPlot&quot;,
                 fluidRow(
                     column(6, selectInput(&quot;x&quot;, label = &quot;Select a feature to visualize&quot;,
                                         choices = as.list(c(&quot;Race&quot;, &quot;title&quot;, &quot;yearsofexperience&quot;, &quot;yearsatcompany&quot;, &quot;Education&quot;))))
                 ),
                 fluidRow(
                     plotOutput(&quot;boxplot&quot;)
                 )
        ),
        tabPanel(&quot;Salary Distribution Plot&quot;,
                 fluidRow(
                     column(12, sliderInput(&quot;range&quot;, &quot;Salary Range:&quot;, min = 10000, max = 2000000, value = 500000,
                                            step = 10000, sep = &quot;&quot;, ticks = FALSE, animate = TRUE)
                     )
                 ),
                 fluidRow(
                     plotOutput(&quot;distributionplot&quot;)
                 )
        )
    )
    
)</code></pre>
<pre><code>## Warning: The select input &quot;company&quot; contains a large number of options; consider
## using server-side selectize for massively improved performance. See the Details
## section of the ?selectizeInput help topic.</code></pre>
<pre class="r"><code>server &lt;- function(input, output) {
    
    output$estimated_salary = renderTable({
        
        salary_df &lt;- data.frame(&quot;Title&quot; = character(), &quot;Salary Estimate&quot; = numeric(), &quot;Confidence interval&quot; = character())
        for(i in 1:length(title_encode$x)) {
            title &lt;- title_encode[i,]$x
            title_idx &lt;- title_encode[i,]$y
        
            race_idx &lt;- filter(race_encode, x == input$race)$y
            education_idx &lt;- filter(education_encode, x == input$education)$y
            company_idx &lt;- filter(company_encode, x == input$company)$y
            experience_idx &lt;- filter(experience_encode, x == input$yearsofexperience)$y
            yearsatcompany_idx &lt;- filter(yearsatcompany_encode, x == input$yearsatcompany)$y
            
            salary &lt;- race_idx * coef[coef == &quot;race&quot;, ]$value + company_idx * coef[coef == &quot;company&quot;, ]$value + education_idx * coef[coef == &quot;education&quot;, ]$value + title_idx * coef[coef == &quot;title&quot;, ]$value + experience_idx * coef[coef == &quot;experience&quot;, ]$value + yearsatcompany_idx * coef[coef == &quot;yearsatcompany&quot;, ]$value + coef[coef == &quot;(Intercept)&quot;, ]$value
            salary_lower &lt;- race_idx * coef[coef == &quot;race&quot;, ]$lower + company_idx * coef[coef == &quot;company&quot;, ]$lower + education_idx * coef[coef == &quot;education&quot;, ]$lower + title_idx * coef[coef == &quot;title&quot;, ]$lower + experience_idx * coef[coef == &quot;experience&quot;, ]$lower + yearsatcompany_idx * coef[coef == &quot;yearsatcompany&quot;, ]$lower + coef[coef == &quot;(Intercept)&quot;, ]$lower
            salary_upper &lt;- race_idx * coef[coef == &quot;race&quot;, ]$upper + company_idx * coef[coef == &quot;company&quot;, ]$upper + education_idx * coef[coef == &quot;education&quot;, ]$upper + title_idx * coef[coef == &quot;title&quot;, ]$upper + experience_idx * coef[coef == &quot;experience&quot;, ]$upper + yearsatcompany_idx * coef[coef == &quot;yearsatcompany&quot;, ]$upper + coef[coef == &quot;(Intercept)&quot;, ]$upper
            
            salary_df &lt;- rbind(salary_df, c(title, round(salary), paste0(round(salary_lower), &quot; - &quot;, round(salary_upper))))
        }
        colnames(salary_df) &lt;- c(&quot;Title&quot;, &quot;Salary Estimate&quot;, &quot;Confidence Interval&quot;)
        salary_df
    })
    
    output$boxplot = renderPlot({
        df$x &lt;- df[, input$x]
        df %&gt;% ggplot(aes(x = as.factor(x), y = basesalary)) +
            geom_boxplot() +
            xlab(sprintf(&quot;%s&quot;, input$x)) +
            ylab(&quot;Base salary&quot;) +
            ggtitle(sprintf(&quot;Distribution of base salary for %s&quot;, input$x)) +
            theme_minimal() +
            theme(plot.title=element_text(hjust=0.5))
    })
    
    output$distributionplot = renderPlot({
        df %&gt;%
            filter(basesalary &lt;= as.numeric(input$range)) %&gt;%
            ggplot(aes(basesalary)) +
            geom_histogram() +
            ggtitle(sprintf(&quot;Distribution for base salary in %s&quot;, input$country2)) +
            ylab(sprintf(&quot;Base salary&quot;)) +
            theme_minimal() +
            theme(plot.title=element_text(hjust=0.5))
    })
}

# Run the application 
# shinyApp(ui = ui, server = server)</code></pre>

</details>
  
  
Demo:

[demo]
