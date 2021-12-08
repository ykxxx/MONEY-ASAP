## Welcome to Your Salary Estimator!

Wanna lead a perfect life without money worries? 🤑🤑🤑

---

As a group of aspiring data scientists, we are highly interested and motivated to embark on our career in tech companies. Before seeking a full-time job, we want to learn more about the salaries from different companies. While some of us have started on our internships, the rest would like to also navigate the data science and STEM career options as possible alternatives. 

### Purpose

We would like to explore more about income and realise our own place predict from analysis of the data. Specifically, we will be analyzing a set of factors that are potentially predictive of the outcome of us earning $100k/yr upon graduation. We will be using a selection of methods to visualise the data as well as reduce the dimension before applying logistic gression, K-nearest neighbor, decision tree, random forest to make predictions. The data would also be presented in a shiny app. Apart from our own money concerns, we would also like to learn about the representativeness of certain socioeconomic groups affect their salary negotiation, and whether our project could help with uncovering the transparency lacking in this dynamic.

Our preferred features include: company, level, title, education level and some other personal backgrounds.

### Data

We used [Kaggle Data Science and STEM income dataset](https://www.kaggle.com/jackogozaly/data-science-and-stem-salaries) for our analysis. This is a dataset scraped from the website of levels.fyi, a site with a visualised, crowdsourced database for transparent leveling charts across companies to help with job negotiation and possibly prevent underpaying for those underrepresented within tech industry.

The original dataset contains 62642 records. We filtered the redundant features and kept `company`, `title`, `totalyearlycompensation`, `location`, `yearsofexperience`, `yearsatcompany`, `basesalary`,`stockgrantvalue`,`bonus`,`gender`,`cityid`,`dmaid`,`Race`,`Education` and corrected datatype.

We wanted to see the base salary of different titles.

[Base salary of different titles](https://github.com/Nancy-dvZhang/MONEY-ASAP/raw/main/images/basesalary_title.png)

Does race matter?

[Base salary of different races](https://github.com/Nancy-dvZhang/MONEY-ASAP/raw/main/images/basesalary_race.png)

What about education?

[Base salary of different level of education](https://github.com/Nancy-dvZhang/MONEY-ASAP/raw/main/images/basesalary_edu.png)

### Our Discovery

[All analysis results]

#### Logistic Regression

#### KNN

#### Decision Tree

#### Randon Forest

### Let's predict!

We built a [shiny app](https://ykxxx.shinyapps.io/predictor/) for users to predict their salaries.

Demo:

[demo]
