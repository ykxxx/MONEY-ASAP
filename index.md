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

<img src="https://github.com/Nancy-dvZhang/MONEY-ASAP/raw/main/images/basesalary_title.png" width="100%"/>  

Does race matter?

<img src="https://github.com/Nancy-dvZhang/MONEY-ASAP/raw/main/images/basesalary_race.png" width="100%"/>  

What about education?

<img src="https://github.com/Nancy-dvZhang/MONEY-ASAP/raw/main/images/basesalary_edu.png" width="100%"/>  

## Our Analysis

<details>
<summary><u>Logistic</u> <b>Regression</b> (<i>click to expand</i>)</summary>
<br>
  
Company and location information is excluded from our multinomial logistic regression model because they are categorical data with so many categories that they obscure our results. First, we fit a full model with all covariates we selected to see the general relationship pattern. We excluded the individuals with NA entries in either Education, Race, and gender to tidy up the data and to make trends more visible. Another reason we decide to drop these entries is that without them, we still have enough data (>20,000) to draw strong prediction power. 

```{r}
df_1 <- filtered_df %>% select(c("title", "yearsofexperience", "yearsatcompany",'gender','cityid','dmaid','Race','Education', 'y')) %>% filter(!is.na(Education) & !is.na(gender) & !is.na(Race))
mod_logistic_whole = glm(y ~ ., data = df_1, family=binomial())
summary(mod_logistic_whole)
```
  
<pre class="r"><code class="hljs">df_1 &lt;- filtered_df %&gt;% select(c(<span class="hljs-string">"title"</span>, <span class="hljs-string">"yearsofexperience"</span>, <span class="hljs-string">"yearsatcompany"</span>,<span class="hljs-string">'gender'</span>,<span class="hljs-string">'cityid'</span>,<span class="hljs-string">'dmaid'</span>,<span class="hljs-string">'Race'</span>,<span class="hljs-string">'Education'</span>, <span class="hljs-string">'y'</span>)) %&gt;% filter(!is.na(Education) &amp; !is.na(gender) &amp; !is.na(Race))
mod_logistic_whole = glm(y ~ ., data = df_1, family=binomial())
summary(mod_logistic_whole)</code></pre>
  
From our full model, we see that all of our covariates provide helpful information with as least one category being significant except gender. We then try a bidirectional step-wise selection procedure to further feature selection.

```{r}
mod_logistic_step <- step(glm(y ~ 1, data = df_1), ~ title + yearsofexperience + yearsatcompany + gender + cityid + dmaid + Race + Education, direction = "both")
summary(mod_logistic_step)
```

The step-wise selection also shows gender to be insignificant, so drop gender. Also, drop entries with 0 in basesalary, they are missing completely at random.

```{r}
df_2 <- filtered_df %>% select(c("basesalary", "title", "yearsofexperience", "yearsatcompany",'cityid','dmaid','Race','Education', 'y')) %>% filter(!is.na(Education)) %>% filter(!is.na(Race)) %>% filter(basesalary!=0)

df_2 <- df_2 %>% select(c("title", "yearsofexperience", "yearsatcompany",'cityid','dmaid','Race','Education', 'y'))
df_2$y <- as.factor(df_2$y)
dim(df_2)
```

Build our final logistic model with the edited dataset.

```{r}
mod_logistic = glm(y ~ ., data = df_2, family=binomial())
summary(mod_logistic)
```

Feature selection done, test accuracy next.

```{r}
set.seed(1)
x <- stratified(df_2, "y", 0.8, keep.rownames = TRUE)
train_set <- x %>% dplyr::select(-rn)
train_index <- as.numeric(x$rn)
test_set <- df_2[-train_index,]
print(dim(train_set))
print(dim(test_set))
head(train_set)
head(test_set)
```

```{r}
mod_logistic_train = glm(y ~ ., data = train_set, family=binomial())
preds_hat <- predict(mod_logistic_train, newdata = test_set, type= "response")
preds <- ifelse(preds_hat>0.5,1,0)
preds <- as.numeric(substring(preds,-1))
confusionMatrix(data = as.factor(preds), reference = as.factor(test_set$y), positive = "1")
```

Our logistic regression model yields an overall accuracy of 0.863 with 95% CI (0.8524, 0.8731) and a very small p-value, which indicates that we did an OK job in predicting basesalary >= 100K. However, our sensitivity is 0.9377 and our specificity is 0.6453, giving a balanced accuracy of 0.7915. This means our logistic regression prediction model is making more Type 1 error (False positive) than type 2 errors.


### Data Manipulation for KNN, Decision Tree, and Random Forest
We want to take a look on how many unqiue values are in our dataset, for categorical feature such as company and feature.
```{r}
filtered_dataset <- filtered_df %>% filter(!is.na(.))
print(length(unique(filtered_dataset$company)))
print(length(unique(filtered_dataset$title)))

```
Since there are too many unqiue company in our dataset, it is better to cluster these companies to reduce the dimensionality of the company feature. Our current approach is to cluster companies based on its size(a.k.a number of time each company appear in the dataset). We re-classify company into 5 classes. 
```{r}
sorted_company <- filtered_dataset %>% count(tolower(company)) %>% arrange(desc(n))
sorted_company
```

```{r}
filtered_dataset <- filtered_dataset %>% mutate(company = tolower(company))
df <- left_join(filtered_dataset, sorted_company, by = c("company" = "tolower(company)"))
df <- df %>% mutate(size_category  = case_when(
  n <= 10 ~ "<10",
  n > 10 & n <= 50 ~ "10~50",
  n > 50 & n <= 100 ~"50~100",
  n > 100 & n <= 500 ~ "100~500",
  n > 500 ~ "500+"

))
df <- df %>% mutate(employee_num = n) %>% select(-n)
head(df)
```
The location feature looks a little bit untidy. So we separate it into city, state, country. 
```{r}
location <- df %>%
  count(location) %>%
  arrange(desc(n)) %>%
  mutate(place = location) %>%
  separate(place, sep = ",",  c("city", "state", "country"))

location[is.na(location)] <- "US"
df <- left_join(df, location, by = "location") %>% select(-location)
head(df)
```

After the manipulation we want to see how exactly the distribution of dataset looks like for different locations
```{r}
df %>% ggplot(aes(size_category, basesalary)) + geom_boxplot()
```
```{r}
df %>% ggplot(aes(country, basesalary)) + geom_boxplot()
```
We want to take a look on the U.S. data specifically
```{r}
df %>% filter(country == "US") %>% ggplot(aes(state, basesalary)) + geom_boxplot()
```
we want to drop missing value and see how the dataset looks like.
```{r}
cleaned_df <- drop_na(df)
dim(cleaned_df)
```
We also need to factorize our variables. 
```{r}
cleaned_df <- cleaned_df %>% mutate(gender = as.factor(gender), size_category = as.factor(size_category), 
                                  state = as.factor(state), country = as.factor(country), y = as.factor(y))
dataset <- cleaned_df %>% select(c("title", "yearsofexperience", "yearsatcompany","gender","cityid","dmaid",
                                  "Race","Education", "y","size_category","state","country"))
sapply(dataset, "class")
```
Now our dataset looks good and we can get started to build up our model!

</details>
  

<details>
<summary><u>K-Nearest Neighbors</u> <b>(KNN)</b> (<i>click to expand</i>)</summary>
<br>
  
```{r}
set.seed(42)
x <- stratified(dataset, "y", 0.8, keep.rownames = TRUE)
train_set <- x %>% dplyr::select(-rn)
train_index <- as.numeric(x$rn)
test_set <- dataset[-train_index,]
print(dim(train_set))
print(dim(test_set))
```

```{r}
set.seed(42)
knn_fit <- train_set %>% knn3(y~., data = ., k = 12)
knn_hat <- predict(knn_fit, newdata = test_set, type = "class")
knn_p <- f_hat2 <- predict(knn_fit, newdata = test_set, k=12)[,2]
confusionMatrix(data = knn_hat, reference = as.factor(test_set$y), positive = "1")
```
                   
</details>
  

<details>
<summary><u>Decision</u> <b>Tree</b> (<i>click to expand</i>)</summary>
<br>
  
We then apply a decision tree to visualize the prediction of the majority class of our outcome (whether one receives >$100k/yr base salary) based within the decision process partitioning based on a combination of variables.
  
We selected a set of covariates that could potentially be predictive of the outcome variable. After combinatory attempts to construct the tree, three combinations resulted in clear partition of majority class in the outcome variable:
  
```{r}
# Create dataset for the trees
dataset_dt <- cleaned_df %>% select(c("title", "yearsofexperience", "yearsatcompany","gender","cityid","dmaid",
                                  "Race","Education", "y","size_category","state","country","employee_num"))
sapply(dataset_dt, class)

# Decision Tree using cityid (proxy for location), years of experience, and years at company
dt_df1 <- dataset_dt %>% dplyr::select(., y, cityid, yearsofexperience, yearsatcompany)
fit1 <- rpart(as.factor(y) ~ ., data = dt_df1)

rpart.plot(fit1)
```
  
In the first tree, the outcome is partitioned by years of experience < 11, city id as a proxy for the importance of locations, and years of experience < 5.
  
```{r}
# Next, Decision Tree using years of experience, and employee number
dt_df2 <- dataset_dt %>% dplyr::select(.,y, yearsofexperience, employee_num)
fit2 <- rpart(as.factor(y) ~ ., data = dt_df2)

rpart.plot(fit2)
```
  
In the second decision tree, if years of experience > 5, we predict the outcome to be >$100k/yr base salary. If not, if employee number of the company is larger than 506, the outcome category is also predicted to be >\$100k/yr base salary. The next divisions are made by years of experience > 3 and emplyee number > 136.
  
```{r}
# Thirdly, Decision Tree using title, company size category, and years of experience
dt_df3 <- dataset_dt %>% dplyr::select(.,y, title, size_category, yearsofexperience)
fit3 <- rpart(as.factor(y) ~ ., data = dt_df3)

rpart.plot(fit3)
```
  
The third tree predicts the outcome by the same cutoffs in years of experience. The rest of the division lies in size category and title.
                           
</details>
  
<details>
<summary><u>Random</u> <b>Forest</b> (<i>click to expand</i>)</summary>
<br>
  
We build a random forest model with 3 features for each tree (~ sqrt level)
```{r}
set.seed(42)
rf_fit <- randomForest(y ~ .-state-country, data = train_set, mtry = 3)
rf_hat <- predict(rf_fit, newdata = test_set, type = "class")
rf_p <- predict(rf_fit, newdata = test_set, type = "prob")[,2]
confusionMatrix(data = rf_hat, reference = as.factor(test_set$y), positive = "1")
```

Now, check the feature importance

```{r}
rf_im <- importance(rf_fit)
as.matrix(rf_im[order(-rf_im[,1]),])
```


### Visualization

Here, we do a visualization using tSNE.

First, to reduce computational burden, we randomly sample 5000 points for visualization. At the same time, since `cityid` and `dmaid` are two strongest factors, we only keep `dmaid` to avoid overfitting.

```{r}
data_sample0 <- dataset[sample(1:nrow(dataset), size = 5000),]
data_sample <- data_sample0[ , -which(colnames(data_sample0) %in% c("cityid"))]
```

```{r}
# Obtain the value of 2-D tSNE embedding
tsne_out <- Rtsne(
 data_sample,
 dims = 2,
 pca = TRUE,
 perplexity = 100,
 check_duplicates = FALSE
)
```

Now, let's see the pattern of different labels

```{r}
rd_rs <- as.data.frame(tsne_out$Y)
rd_rs$Class<- data_sample$y
# first obtain the color
length(unique(rd_rs$Class))
mainPalette <- rainbow(9)
ggplot(rd_rs, aes(x=V1, y=V2, color=Class)) +
  geom_point(size=1.25) +
  labs(title = "t-SNE",
       x = "t-SNE1",
       y = "t-SNE2") +
  theme(plot.title = element_text(hjust = 0.5))
```

Let's see the pattern of the strongest factor `dmaid`

```{r}
rd_rs <- as.data.frame(tsne_out$Y)
rd_rs$Class<- data_sample$dmaid
# first obtain the color
length(unique(rd_rs$Class))
mainPalette <- rainbow(9)
ggplot(rd_rs, aes(x=V1, y=V2, color=Class)) +
  geom_point(size=1.25) +
  labs(title = "t-SNE",
       x = "t-SNE1",
       y = "t-SNE2") +
  theme(plot.title = element_text(hjust = 0.5))
```
                   
</details>

<details>
<summary><u>Linear</u> <b>Regression</b> (<i>click to expand</i>)</summary>
<br>
  
We also wanted to fit the Dataset into a linear regression model so that we will be able to make predictions on base salary given information about a person's relevant features. So in this analysis, we explored the performance of different feature engineering and encoding methods, as well as the prediction performance of several linear regression model. We will choose the best model to make our base salary predictor in the next section.


## Explore Dataset

```{r}
dataframe = read.csv("Levels_Fyi_Salary_Data.csv")
head(dataframe)
```

```{r}
filtered_df <- dataframe %>% 
  select(c("company", "title", "totalyearlycompensation", "location", "yearsofexperience", "yearsatcompany", 
'basesalary','stockgrantvalue','bonus','gender','cityid','dmaid','Race','Education')) %>% 
  mutate(company = as.factor(company), title <- as.factor(title), Race <- as.factor(Race), Education <- as.factor(Education)) %>%
  filter(!is.na(Education) & !is.na(gender) & !is.na(Race))

head(filtered_df)
```

Since the location in this datafram is encoded as City, State, Country (except for locations in the US, where it is encoded as City, State), we would like to split this into 3 separate features and take a look into the distribution of countries. 

```{r}
location <- filtered_df %>%
  count(location) %>%
  arrange(desc(n)) %>%
  mutate(place = location) %>%
  separate(place, sep = ",",  c("city", "state", "country"))

location[is.na(location)] = "US"
```

Join the location table with our original dataset.

```{r}
joined_df <- filtered_df %>%
  left_join(location, by = "location")

head(joined_df)
```

After ploting the base salary for different countries, we can see that there is a large discrapencies between each countries.

```{r}
joined_df %>% 
  ggplot(aes(country,basesalary))  + 
  geom_boxplot() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  ylim(0,400000)
```

If we list the number of data we have for each countries, we can also see that the dataset is very imbalanced -- the majoriety of data comes from the US, where as other countries only contribute a very small amount of data, this can be a major problem for machine learning models, since there may not be enough data for the model to learn the representation for other countries properly. Therefore, we decided to filter only data from the US, and train the linear regression model as a base salary predictor for the US.

```{r}
joined_df %>%
  count(country) %>%
  arrange(desc(n))
```

```{r}
us_df <- joined_df %>%
  filter(country == "US")

head(us_df)
```


### Feature Engineering

Next we would like to proceed to the feature engineering stage. Since we are building the linear regression model to use it as the base salary predictor for our Shiny App, we can only include features that we want users to input for their salary prediction. This means that we cannot use feature such as `dmaid` in the model since we don't know what this feature represent. So after looking at our EDA plots, we decided to choose 6 features: `title`, `company`, `yearsatcompany`, `yearsofexperience`, `Education`, and `Race` as our features. 

Considering the ease of calculation for our Shiny App Predictor, we also decided to use a categorical encoding for each feature instead of one-hot encoding, even though empirically one-hot encoding may yield better results. 

To do the feature engineering for `title`, we plot the average base salary for each title again. We can see that there is a clear difference in average base salary for different title.

```{r}
us_df %>% 
  drop_na() %>%
  mutate(company = str_to_title(company)) %>%
  select(c("company", "title", "location", "yearsofexperience", "yearsatcompany", 
'basesalary','gender','cityid','dmaid','Race','Education'))

```

```{r}
us_df %>% 
  ggplot(aes(title,basesalary))  + 
  geom_boxplot() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  ylim(0,400000)
```

In order to make sure that our title encoding can capture this pattern of differences, we decided to first encode each title category as their average base salary, and later we would also normalize this encoding value to range between [0, 1].

```{r}
title_mean <- us_df %>%
  group_by(title) %>%
  summarise(title_encode = mean(basesalary)/10000) %>%
  arrange(desc(title_encode))

title_mean
```

We repeat the same process for the other 5 featues.

```{r}
us_df %>% 
  filter(!is.na(Race)) %>%
  ggplot(aes(Race,basesalary))  + 
  geom_boxplot() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  ylim(0,400000)
```

```{r}
race_mean <- us_df %>%
  group_by(Race) %>%
  summarise(race_encode = mean(basesalary)/10000) %>%
  arrange(desc(race_encode))

race_mean
```

```{r}
us_df %>% 
  filter(!is.na(Education)) %>%
  ggplot(aes(Education,basesalary))  + 
  geom_boxplot() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  ylim(0,400000)
```

```{r}
education_mean <- us_df %>%
  group_by(Education) %>%
  summarise(education_encode = mean(basesalary)/10000) %>%
  arrange(desc(education_encode))

education_mean
```

```{r}
us_df %>% 
  filter(!is.na(yearsofexperience)) %>%
  ggplot(aes(as.factor(yearsofexperience),basesalary))  + 
  geom_boxplot() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  ylim(0,400000)
```

```{r}
experience_mean <- us_df %>%
  group_by(yearsofexperience) %>%
  summarise(experience_encode = mean(basesalary)/10000) %>%
  arrange(desc(experience_encode))

experience_mean
```

```{r}
us_df %>% 
  filter(!is.na(yearsatcompany)) %>%
  ggplot(aes(as.factor(yearsatcompany),basesalary))  + 
  geom_boxplot() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  ylim(0,400000)
```

```{r}
yearatcompany_mean <- us_df %>%
  group_by(yearsatcompany) %>%
  summarise(yearatcompany_encode = mean(basesalary)/10000) %>%
  arrange(desc(yearatcompany_encode))

yearatcompany_mean
```

For the feature `company`, we know that this dataset includes data from 1633 distinct conpaies, so we think it would probably be better it we convert `company` into new new feature with less levels. To do this, we first take a look into how the `company` feature distribution looks like.

```{r}
company <- us_df %>%
  count(company) %>%
  arrange(desc(n))

head(company)
```

We also plot this distribution in a histrogram.

```{r}
company %>%
  filter(n <= 100) %>%
  ggplot(aes(n)) +
  geom_histogram(bins = 20)
```

After taking a look into the specific data, we see that the number of appearance a company has in this dataset, `n`, actually highly correlates with the actual company size. For instance, the major tech companies like Amazon, Micorsoft, Google, Facebook, and Apple ranks top 5 in the tabke above, and all has over 1000 data entries. So we decided to use this `n` create a new feature called `company_size` and use this feature for our model training. 

```{r}
company_size <- company %>%
  mutate(company_size = case_when(
    n <= 3 ~ "1",
    n <= 10 ~ "3 - 10",
    n <= 50 ~ "11 - 50",
    n <= 100 ~ "51 - 100",
    n <= 500 ~ "101 - 500",
    TRUE ~ ">500"
  ))
```

```{r}
us_df <- us_df %>%
  left_join(company_size, by = "company")

head(us_df)
```

We also repeat the same process for feature engineering.

```{r}
us_df %>% 
  filter(!is.na(company_size)) %>%
  ggplot(aes(company_size,basesalary))  + 
  geom_boxplot() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  ylim(0,400000)
```

```{r}
company_mean <- us_df %>%
  group_by(company_size) %>%
  summarise(company_encode = mean(basesalary)/10000) %>%
  arrange(desc(company_encode))

company_mean
```

To create the final dataframe for our model, we need to join all encoded feature together and normalize these feature to range between [0, 1] using the `min_max_norm` function.

```{r}
min_max_norm <- function(x) {
    (x - min(x)) / (max(x) - min(x))
  }
```

```{r}
lm_us_df<- us_df %>%
  left_join(company_mean, by = "company_size") %>%
  left_join(title_mean, by = "title") %>%
  left_join(education_mean, by = "Education") %>%
  left_join(race_mean, by = "Race") %>%
  left_join(experience_mean, by = "yearsofexperience") %>%
  left_join(yearatcompany_mean, by = "yearsatcompany") %>%
  mutate(company = min_max_norm(company_encode), title = min_max_norm(title_encode), education = min_max_norm(education_encode), race = min_max_norm(race_encode), experience = min_max_norm(experience_encode), yearsatcompany = min_max_norm(yearatcompany_encode)) %>%
  select(basesalary, company, title, education, race, experience, yearsatcompany)

head(lm_us_df)
```


### Saving cleaned data and encoded features

We save the cleaned dataframe into a .csv file and also save all the encoding data so that we can use them for reference when we later calculate the predicted base salary in our Shiny App Predictor.

```{r}
write.csv(x=us_df, file="project/clean_us_salary_data.csv", row.names = FALSE)
```

```{r}
company_encode <- company_size %>%
  left_join(company_mean, by = "company_size") %>%
  mutate(x = company, y = min_max_norm(company_encode)) %>%
  select(x, y, company_size)

write.csv(x=company_encode, file="project/company_encode.csv", row.names = FALSE)

head(company_encode)
```

```{r}
education_encode <- education_mean %>%
  mutate(x = Education, y = min_max_norm(education_encode)) %>%
  select(x, y)

write.csv(x=education_encode, file="project/education_encode.csv", row.names = FALSE)

head(education_encode)
```

```{r}
title_encode <- title_mean %>%
  mutate(x = title, y = min_max_norm(title_encode)) %>%
  select(x, y)

write.csv(x=title_encode, file="project/title_encode.csv", row.names = FALSE)

head(title_encode)
```

```{r}
yearsatcompany_encode <- yearatcompany_mean %>%
  mutate(x = yearsatcompany, y = min_max_norm(yearatcompany_encode)) %>%
  select(x, y)

write.csv(x=yearsatcompany_encode, file="project/yearsatcompany_encode.csv", row.names = FALSE)

head(yearsatcompany_encode)
```

```{r}
race_encode <- race_mean %>%
  mutate(x = Race, y = min_max_norm(race_encode)) %>%
  select(x, y)

write.csv(x=race_encode, file="project/race_encode.csv", row.names = FALSE)

head(race_encode)
```

```{r}
experience_encode <- experience_mean %>%
  mutate(x = yearsofexperience, y = min_max_norm(experience_encode)) %>%
  select(x, y)

write.csv(x=experience_encode, file="project/experience_encode.csv", row.names = FALSE)

head(experience_encode)
```


### Training the Linear Regression model

To train our linear regression model, we first need to split the dataset into training and testing.

```{r}
set.seed(1)
idx <- sample.int(n = nrow(lm_us_df), size = floor(0.8 * nrow(lm_us_df)), replace = FALSE)

train <- lm_us_df[idx, ]
test <- lm_us_df[-idx, ]

dim(train)
dim(test)
```

Fit the training dataset into the linear regression model.

```{r}
lm1 <- lm(basesalary ~ ., data = train)
summary(lm1)
```

We can see that all the features are significant, this suggests that they are all important and contribute to the final prediction of base salary. So we now save these coefficient for our base salary predictor.

```{r}
lm_coef <- summary(lm1)$coefficients
lm_coef_df <- data.frame(coef = rownames(lm_coef), value = lm_coef[, 1], std = lm_coef[, 2], lower = lm_coef[, 1] - lm_coef[, 2], upper = lm_coef[, 1] + lm_coef[, 2])
write.csv(lm_coef_df, file = "project/lm_coef.csv", row.names = FALSE)
```

To evaluate model performance, we choose to calculate its RMSE.

```{r}
RMSE <- function(true_ratings, predicted_ratings){
    sqrt(mean((true_ratings - predicted_ratings)^2))
}
```

The RMSE for the test dataset prediction is relatively large, however, considering the noise that naturally exists in this dataset, where people with the exact same features is likely to have different base salary, a large testing RMSE doesn't necessarily suggest that our model is performing bad, and we would need further analysis to fairly evaluate our model performance.

```{r}
lm_pred <- predict(lm1, newdata = test)
lm_rmse <- RMSE(lm_pred, test$basesalary)
lm_rmse
```

By comparing our model prediction and the actual base salary in the testing dataset, we can see that while not accurate, our model is able to capture the general trend of base salary level. This means that the prediction of out model can at least provide some reference for a person's potential base salary based on their input.

```{r}
lm_pred[1:10]
test$basesalary[1:10]
```
                   
</details>
  

<details>
<summary><u>Optimizing Model Performance - </u> <b>One-hot Encoding and Regularization</b> (<i>click to expand</i>)</summary>
<br>

To better evaluate our previous model performance, we also want to explore whether we can optimize the previous simple Linear Regression model through one-hot encoding and regularization. To do this, we first need to prepare the one-hot encoded dataset.

```{r}
library(caret)

df_one_hot <- us_df %>%
  select(title, yearsofexperience, yearsatcompany, basesalary, Education, company_size)

dmy <- dummyVars(" ~ .", data = df_one_hot)
df_one_hot <- data.frame(predict(dmy, newdata = df_one_hot))
str(df_one_hot)
```

Split training and testing dataset.

```{r}
train_one_hot <- df_one_hot[idx, ]
test_one_hot <- df_one_hot[-idx, ]

dim(train_one_hot)
dim(test_one_hot)
```

Fit a linear regression model using the one-hot encoded dataset.

```{r}
lm2 <- lm(basesalary ~ ., data = train_one_hot)
summary(lm2)
```

Evaluting model performance using RMSE on testing dataset. The RMSE is slightly lower than our previous linear regression model, which suggests that one-hot encoding do yield better results. However, since the difference is pretty small, we would still use the weight from our previous model for the ease of calculation.

```{r}
lm2_pred <- predict(lm2, newdata = test_one_hot)
lm_rmse <- RMSE(lm2_pred, test_one_hot$basesalary)
lm_rmse
```

To explore the effect of regularization, we also fit the one-hot encoded data into a ridge regression model.

```{r}
library(glmnet)

x = as.matrix(subset(train_one_hot, select=-c(basesalary)))
y_train = train_one_hot$basesalary
x_test = as.matrix(subset(test_one_hot, select=-c(basesalary)))
y_test = train_one_hot$basesalary

ridge_reg = glmnet(x, y_train, alpha = 1, family = 'gaussian', lambda = 0.8)
ridge_reg
```

The RMSE is higher for ridge regression, so we would not choose this model.

```{r}
pred <- predict(ridge_reg, newx = x_test)[, 1]
rmse_ridge <- RMSE(pred, y_test)
rmse_ridge
```

Overall, we think our first simple linear regression gives a reasonably good prediction on base salary, so we decided to use its weight to build our final salary predictor shiny app.

</details>
  

### Our Discovery

[比较model]

### Let's predict!

We built a [shiny app](https://ykxxx.shinyapps.io/predictor/) for users to predict their salaries.

<details>
<summary><u>Shiny</u> <b>APP</b> (<i>click to expand</i>)</summary>
<br>

We implemented 3 main features in our Shiny App, which are embedded in 3 seperate tabs.

### Estimator

User can input their information for each of the 5 features in the drop-down bar, and our salary estimator will return a table of salary estimation as well as the confidence intercal for each of the 15 titles.

### Boxplot

User can view the boxplot of base salary distribution for each of the 6 features

### Salary distribution plot

User can select a base salary upper bound, and view the distribution of salary.

```{r}
library(shiny)
library(tidyverse)
library(ggplot2)

# library(rsconnect)
# rsconnect::deployApp('/Users/kexinyang/Desktop/BST 260/Project/notebook.Rmd')

df <- read.csv("project/clean_us_salary_data.csv")
coef <- read.csv("project/lm_coef.csv")
race_encode <- read.csv("project/race_encode.csv")
company_encode <- read.csv("project/company_encode.csv")
education_encode <- read.csv("project/education_encode.csv")
title_encode <- read.csv("project/title_encode.csv")
experience_encode <- read.csv("project/experience_encode.csv")
yearsatcompany_encode <- read.csv("project/yearsatcompany_encode.csv")

ui <- fluidPage(
    
    theme = shinythemes::shinytheme("flatly"),

    titlePanel("Salary Estimator :)"),
    
    tabsetPanel(
        tabPanel("Estimator",
                 sidebarLayout(
                     sidebarPanel("Choose your features:", 
                        fluidRow(selectInput("race", label = "Select a race", selected = "Asian",
                                           choices = race_encode$x)),
                        fluidRow(selectInput("company", label = "Select a company", selected = "Amazon",
                                           choices = company_encode$x)),
                        fluidRow(selectInput("education", label = "Select a education",
                                           choices = education_encode$x)),
                        fluidRow(selectInput("yearsofexperience", label = "Select a years of experience", selected = "1",
                                           choices = experience_encode$x)),
                        fluidRow(selectInput("yearsatcompany", label = "Select a years at company", selected = "0",
                                           choices = yearsatcompany_encode$x))
                     ),
                     mainPanel(
                         tableOutput("estimated_salary")
                     )
                 ),
                 ),
        tabPanel("BoxPlot",
                 fluidRow(
                     column(6, selectInput("x", label = "Select a feature to visualize",
                                         choices = as.list(c("Race", "title", "yearsofexperience", "yearsatcompany", "Education"))))
                 ),
                 fluidRow(
                     plotOutput("boxplot")
                 )
        ),
        tabPanel("Salary Distribution Plot",
                 fluidRow(
                     column(12, sliderInput("range", "Salary Range:", min = 10000, max = 2000000, value = 500000,
                                            step = 10000, sep = "", ticks = FALSE, animate = TRUE)
                     )
                 ),
                 fluidRow(
                     plotOutput("distributionplot")
                 )
        )
    )
    
)

server <- function(input, output) {
    
    output$estimated_salary = renderTable({
        
        salary_df <- data.frame("Title" = character(), "Salary Estimate" = numeric(), "Confidence interval" = character())
        for(i in 1:length(title_encode$x)) {
            title <- title_encode[i,]$x
            title_idx <- title_encode[i,]$y
        
            race_idx <- filter(race_encode, x == input$race)$y
            education_idx <- filter(education_encode, x == input$education)$y
            company_idx <- filter(company_encode, x == input$company)$y
            experience_idx <- filter(experience_encode, x == input$yearsofexperience)$y
            yearsatcompany_idx <- filter(yearsatcompany_encode, x == input$yearsatcompany)$y
            
            salary <- race_idx * coef[coef == "race", ]$value + company_idx * coef[coef == "company", ]$value + education_idx * coef[coef == "education", ]$value + title_idx * coef[coef == "title", ]$value + experience_idx * coef[coef == "experience", ]$value + yearsatcompany_idx * coef[coef == "yearsatcompany", ]$value + coef[coef == "(Intercept)", ]$value
            salary_lower <- race_idx * coef[coef == "race", ]$lower + company_idx * coef[coef == "company", ]$lower + education_idx * coef[coef == "education", ]$lower + title_idx * coef[coef == "title", ]$lower + experience_idx * coef[coef == "experience", ]$lower + yearsatcompany_idx * coef[coef == "yearsatcompany", ]$lower + coef[coef == "(Intercept)", ]$lower
            salary_upper <- race_idx * coef[coef == "race", ]$upper + company_idx * coef[coef == "company", ]$upper + education_idx * coef[coef == "education", ]$upper + title_idx * coef[coef == "title", ]$upper + experience_idx * coef[coef == "experience", ]$upper + yearsatcompany_idx * coef[coef == "yearsatcompany", ]$upper + coef[coef == "(Intercept)", ]$upper
            
            salary_df <- rbind(salary_df, c(title, round(salary), paste0(round(salary_lower), " - ", round(salary_upper))))
        }
        colnames(salary_df) <- c("Title", "Salary Estimate", "Confidence Interval")
        salary_df
    })
    
    output$boxplot = renderPlot({
        df$x <- df[, input$x]
        df %>% ggplot(aes(x = as.factor(x), y = basesalary)) +
            geom_boxplot() +
            xlab(sprintf("%s", input$x)) +
            ylab("Base salary") +
            ggtitle(sprintf("Distribution of base salary for %s", input$x)) +
            theme_minimal() +
            theme(plot.title=element_text(hjust=0.5))
    })
    
    output$distributionplot = renderPlot({
        df %>%
            filter(basesalary <= as.numeric(input$range)) %>%
            ggplot(aes(basesalary)) +
            geom_histogram() +
            ggtitle(sprintf("Distribution for base salary in %s", input$country2)) +
            ylab(sprintf("Base salary")) +
            theme_minimal() +
            theme(plot.title=element_text(hjust=0.5))
    })
}

# Run the application 
shinyApp(ui = ui, server = server)

```

</details>
  
  
Demo:

[demo]
