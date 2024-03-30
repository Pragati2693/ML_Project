# ML_Project
## Crowd Funding

Machine Learning in Practice - Assessment

**Context**

Crowdfunding is the practice of raising money for a project from the people around the globe. Initial coin offerings (ICOs) are a popular way to raise funds for products and services usually related to cryptocurrency.
While it is true that crowdfunding is the most popular way to raise fund, there is a significant portion of the companies who are unable to reach the goal. In fact, on KickStarter, only about 35 percent of the total projects have raised successful fundings in the past (Bansal,2019). This finding raises an important question as which project would be able to succeed, what are the important factors the one should consider beforehand in order to make a project reach its goal and how to identify those factors.

**About Dataset and Problem Statement**

Given dataset contains a total of 2767 records and it contain below information about each of those 2767 companies:
ID, Success, brandSlogan, hasVideo, rating, priceUSD, countryRegion, startDate, endDate, teamSize, hasGithub, hasReddit, platform, coinNum, minInvestment, distributedPrecentage

The explanation for all above column names is provided in the assessment Coursework, thus leaving those explanation here.
This coursework will try to find the answers of the below question by the end this project:
What are the factors which are/is most important to consider in order to achieve the goal/fund for the success of the company.
To what extend those factors can be manipulated for success of the company
To predict the success of the company by applying Machine learning algorithm and selecting the best the algorithm which can give the highest accuracy rate in predicting a company’s success. 

**Pre-processing of the dataset**

The given dataset contains 1267 rows and 16 columns. There are a total of 405 missing values in columns as below:
Columns	Total Missing value count
priceUSD	180
countryRegion	71
teamSize	154
 
![image](https://github.com/Pragati2693/ML_Project/assets/68961996/c630f909-d449-48ee-901f-9fb2c86a8bb2)
 
**Fig: 1.1. Missing values**

Lets check the type of the values in each column: 
ID: has no missing/wrong values and is of numeric type
Success: has no missing/wrong value, but needs to converted into categorical column therefore using “as.factor()” column is converted to categorical column
brandSlogan: no missing value and of character type
hasVideo, hasGithuib, hasReddit, minInvestment: has no missing/wrong value, but needs to converted into categorical column therefore using “as.factor()” column is converted to categorical column.
Rating: All the values are between 0-5, no missing or negative value and no outliers present.
startDate: value start from 10/01/2010 till 06/06/2020, thus needs to be converted to date format.
endDate: value start from 31/05/2016 till 10/08/2020, thus needs to be converted to date format.
Duration: new column added obtained from the difference between endDate and startDate. Duration has outlier (as below) and thus need to be removed.


 ![image](https://github.com/Pragati2693/ML_Project/assets/68961996/e9c849f1-f0bb-4d8b-bf4d-8e1ea6099a5c)

**Fig: 1.2: Outliers in Duration Column**

From the above figure, it is clear that are some values which is negative, which cannot be possible because start date should never be greater than end date and also there is one value which is very high, greater than 3000 days. Thus, a total of 12 rows needs to be removed (11 neg values and one value greater than 3000).

After removing all the outliers and replacing wrong values, we are left with 2412 rows.
countryRegion: there are 71 missing values and other value are written in different style.
Converting all values in capitals letter using toupper() function and then finding the unique values.
There are 117 unique values excluding missing values but, MÃ‰XICO is same is MAXICO and CURAÃ‡AO is same is CURACAO, thus replacing these 2 values with the real name there are 115 unique values.
Platform: there are 6 blank values and a lot of duplicate values as below:
	"Ethereum", "Ethereum   ","Ethereum ","Ethereum  " ,"Ethereum     ","Ethereum    "," Ethereum","Ethererum" are same
	"pow/pos", "PoW/PoS","POS + POW","POS,POW" are same, 
	"Neo", "NEO"," NEO" are same
	"Separate Blockchain", "Separate Blockchain ","Separate blockchain", "Separate Blockchain  " are same
	"X11 blockchain" and "X11" are same 
	"X13", "x13 " are same
	"Tron    ","Tron ","TRON", "Tron" are same
	"EOS", "Eos" are same
	"Waves", "WAVES" are same
	"DPOS","DPoS" are same
	"Nem", "NEM" are same
	"Pivx", "PivX" are same
	"Komodo",".Komodo" are same
Values with space: 
	"Neblio "
	"TTchain  "
	"MAHRA platform "

There are a total of 6 cells of Platform column which only has a space “ “ so we need to replace it with NA value.


After relacing all above values with correct values, we got 101 unique values excluding missing values.

priceUSD: 180 missing values and the values lie between 0-39384.
Ideally, the price of bitcoin generated which is to be issued by the company cannot have 0 value, thus all the rows with 0 as price and outliers has to be removed.
There are 152 rows which has $0 price and an outlier having more than $20000 needs to be removed.
Outlier for PriceUSD column

 ![image](https://github.com/Pragati2693/ML_Project/assets/68961996/32819a61-b9cd-4d4a-a5e3-988e470416a8)

**Fig: 1.3: Outliers in Price Column**

teamSize: There are 154 missing values and other values lie between 1 – 75. 

![image](https://github.com/Pragati2693/ML_Project/assets/68961996/dec830d6-6272-4be4-a919-739788abb381)

**Fig: 1.4: Outliers in TeamSize Column**

 	Not removing outliers because from the observation all the team size value greater than 50 has 100% success rate, so there might be some relation, thus leaving this column.

coinNum: Has no missing values but its value lies between 12 to 2.2 * 10^16, thus it indicates that it may have outliers.
Using function: boxplot.stats(tdata$coinNum)$out, gives 395 outliers but removing all these rows might result in loss of data, thus removing only the highest value which is greater than 1.5* 10^15.
 
![image](https://github.com/Pragati2693/ML_Project/assets/68961996/1e2377c5-909f-43e6-bed1-be36882a1ede)

**Fig: 1.5: Outliers in coinNum Column**

distributedPercentage: No missing values but a lot of Outlier company cannot distributes

 ![image](https://github.com/Pragati2693/ML_Project/assets/68961996/9f232a67-77f2-4b46-9cbb-cf8d0a63fdeb)

**Fig: 1.6: Outliers in Distribution%**
Since logically, a company cannot distribute more than 100%, thus we need to remove all the rows which has more than 100 values in the dataset and there are 10 rows which has more than 100 value.

**After removing all the outliers and replacing wrong values, we are left with 2412 observations, in which 203 observations has missing values.**

### IMPUTATION

Imputing the data with random forest method, because it can impute categorical as well as non-categorical data and polyreg method gave error that “countryRegion” categories exceeded more than 50.
imp_data1<-mice(Cdata, m = 5, method = "rf")
Cdata<-complete(imp_data1)
sum(is.na(Cdata)

And now no null value is present in the cleaned final dataset.

#### Understanding the Dataset

Now we have a total of 2412 observations and 17 columns. Out of 2412 records or companies only 916 companies (38% of the companies) were successfully able to raised the given amount of fund during a particular interval of time while remaining 1496 (62%) didn’t succeeded. 

***Understanding the success rate w.r.t country***
There are a total of 115 countries and the top 10 countries which account for maximum number to companies are:
Country Name	Total Number of Companies
USA	273
SINGAPORE	268
UK	245
ESTONIA	171
SWITZERLAND	134
RUSSIA	131
CAYMAN ISLANDS	58
GERMANY	58
NETHERLANDS	55
MALTA	51
 
 ![image](https://github.com/Pragati2693/ML_Project/assets/68961996/7270102d-5144-481a-a1e8-a2de05a8a5c9)

**Fig: 2.1: Total number of companies verses Country**

From the above fig, maximum number of companies are from USA and the maximum number of the companies succeeded in raising fund are from UK and the ratio of success/not success is highest for companies from Cayman Islands.
Understanding the success rate w.r.t Platform

![image](https://github.com/Pragati2693/ML_Project/assets/68961996/f81a3f3b-bff5-4fdb-a827-bacca788f44c)

Most of companies, approx. 87% of the companies, uses “Ethereum” platform and the top 5 platforms and frequency of its use is as below: 
 
Because the count of the rest of the platforms except for Ethereum are negligible, thus calculating the success rate of companies using Ethereum platform we got that only 38% of the companies using Ethereum platform succeeded.
Success rate of companies using “Ethereum” Platform is 38%

 ![image](https://github.com/Pragati2693/ML_Project/assets/68961996/f94786ae-71c0-46ee-bb9d-aeda57e4dcc0)

**Fig: 2.2: Success percentage of companies using Ethereum Platform**

***Understanding the success rate w.r.t companies having Video***
There are 1771 (73%) of the companies which has video and 641 (27%) companies did not have.
Only 42% of the companies having Video succeeded and 25% of the companies not having video succeeded. Clearly the ratio of the companies having video has more success rate than those who do not have.

 ![image](https://github.com/Pragati2693/ML_Project/assets/68961996/7eedeaf4-5f87-42ee-8e7d-efa7523fae56)

**Fig: 2.3: Total number of companies (Success and not Success) verses hasVideo**

Understanding the success rate w.r.t companies having GitHub
There are 1396 companies (58%) which has GitHub account and 1016 companies (42%) didn’t have. 43% of the companies having GitHub succeeded while 30% of the companies didn’t have GitHub account succeeded, thus the ratio of the companies having GitHub account has more success rate than those who do not have.
 
**Fig: 2.4: Total number of companies (Success and not Success) verses hasGithub**

Understanding the success rate w.r.t companies having Reddit
There are 1553 companies (64%) which has Reddit account and 859 companies (37%) didn’t have. 44% of the companies having Reddit succeeded while 27% of the companies didn’t have Reddit account succeeded, thus the ratio of the companies having Reddit account has more success rate than those who do not have.
 
**Fig: 2.5: Total number of companies (Success and not Success) verses hasReddit**

Understanding the success rate w.r.t companies require Minimum Investment 
There are 1100 companies (46%) which require MinimumInvestment and 1312 companies (54%) didn’t require. 40% of the companies require MinimumInvestment succeeded while 36% of the companies didn’t require MinimumInvestment succeeded, thus the ratio of the companies require MinimumInvestment has more success rate than those who do not require.
 
**Fig: 2.6: Total number of companies (Success and not Success) verses minInvestment required**

### Slogan:
Slogan column contains a mini-mission statement of every company, and the words like “decentr”, “cryptocurrency”, “crypto” etc. are used is most of the company’s Slogan.
 
**Fig: 2.7: WordCloud on Slogan column**

Finding the Corelation between all the variables possible with the help of Corrplot
To draw corrplot all the variables need to be converted to numeric value and into 0 or 1 form. Since slogan cannot be converted to 0/1 so we removed it. Also, countryRegion has more than 50 countries thus removing countryRegion column as well. Platform column has 87% of its value as Ethereum thus, to make it simple the values is converted as 1 if platform is Ethereum type and 0 if other than Ethereum. ID is also removed as it has no significance. StartDate and EndDate is also removed and Duration column is used.
The below correlation plot, Success of the company has positive correlation with Rating, TeamSize, HasReddit, hasGitHub and negative correlation with duration.
But the below graph doesn’t give very clear picture of relation between success of the company and other variables.
Correlation plot between the columns 
 
**Fig: 2.8: Correlation between each columns**

So, lets verify the relation between variables using Spearsman’s test.
Verifying only the success column, attached pdf of Spearrsman’s test for all variables

<img width="457" alt="image" src="https://github.com/Pragati2693/ML_Project/assets/68961996/5b6a5df9-a398-4c76-8894-78f1ce4eb9c5">
	

**. Correlation is significant at the 0.01 level (2-tailed).
*. Correlation is significant at the 0.05 level (2-tailed).

From the above test, all the columns except MinInvestment, Distribution% and Platform do affect the result (success) of the company at CI of 99%.



### Scaling and Fitting for applying Machine learning Algorithms

There are 2 method for scaling the columns – Standardization and Normalisation but more accuracy was achieved when applied in KNN method with standardization than normalization, so dataset scaled with standardization is used for all other Algorithms.
Standardization of column: Rating , PriceUSD, teamSize, coinNum, Distribution% and Duration as per below function

<img width="276" alt="image" src="https://github.com/Pragati2693/ML_Project/assets/68961996/834eb7e0-c686-43a9-a0fb-f878819ab452">




#### Categorising CountryRegion

For applying ML algorithm, we categorise the country on the bais of success rate, for example the country for which success rate is between 0 to 10% will be replaced with name C_0_10 and the country for which success rate is between 71 to 100% is replaced with name C_71_100, as below:
C_0_10<- c("AFGHANISTAN","ANDORRA","ARGENTINA","ARMENIA","BARBADOS","BELARUS", "BOSNIA AND HERZEGOVINA", "CAMBODIA","CHILE", "CONGO","CURACAO","DOMINICAN REPUBLIC","ECUADOR","FRENCH,POLYNESIA","GHANA","HUNGARY","ICELAND","KAZAKHSTAN","KUWAIT","KYRGYZSTAN", "MONGOLIA","NEW CALEDONIA","PHILIPPINES","PUERTO RICO","SAINT VINCENT AND THE GRENADINES","SAUDI ARABIA","TIMOR-LESTE","TUNISIA","VANUATU","VENEZUELA")
C_11_20<- c("CROATIA","GREECE","IRELAND","MARSHALLISLANDS","MAURITIUS","NIGERIA", "PORTUGAL", "SERBIA","SPAIN")
C_21_30<- c("BELGIUM","BRAZIL","COSTA RICA","CZECH REPUBLIC","DENMARK", "INDIA", "INDONESIA","ISRAEL","ITALY","MACEDONIA","MALAYSIA","NORWAY","PANAMA","POLAND","ROMANIA","SWEDEN","THAILAND","TURKEY","UKRAINE","USA")
C_31_40<- c("ANGUILLA","AUSTRALIA","BAHAMAS","CANADA","CHINA","FRANCE","GERMANY",  "LATVIA","LUXEMBOURG","MALTA","NETHERLANDS","RUSSIA","SAINT KITTS AND NEVIS", "SEYCHELLES","SOUTH KOREA","UK")
C_41_50<-c("AUSTRIA","BRITISH VIRGIN ISLANDS","CAYMAN ISLANDS","COLOMBIA", "CYPRUS","ESTONIA","FINLAND","GIBRALTAR","ISLE OF MAN","JAPAN","LITHUANIA","MEXICO", "PAKISTAN","PERU","SINGAPORE","SLOVENIA","SWITZERLAND","UNITED ARAB EMIRATES")
C_51_60<-c("BELIZE","BULGARIA","SOUTH AFRICA","VIETNAM")
C_61_70<-c("BERMUDA","GEORGIA","NEW ZEALAND","LIECHTENSTEIN")
C_71_100<-c("EGYPT","GUINEA-BISSAU","MONTENEGRO","SAMOA","SLOVAKIA", "TANZANIA", "ZIMBABWE")
#### Replacing values
Pdata$countryRegion[Pdata$countryRegion %in% C_0_10]<-"0_10"
Pdata$countryRegion[Pdata$countryRegion %in% C_11_20]<-"11_20"
Pdata$countryRegion[Pdata$countryRegion %in% C_21_30]<-"21_30"
Pdata$countryRegion[Pdata$countryRegion %in% C_31_40]<-"31_40"
Pdata$countryRegion[Pdata$countryRegion %in% C_41_50]<-"41_50"
Pdata$countryRegion[Pdata$countryRegion %in% C_51_60]<-"51_60"
Pdata$countryRegion[Pdata$countryRegion %in% C_61_70]<-"61_70"
Pdata$countryRegion[Pdata$countryRegion %in% C_71_100]<-"71_100"

#### Categorising Platform

Categorizing platform on the basis of Ethereum or not-Ethereum
Mdata$platform<-ifelse(Mdata$platform == "Ethereum",1,0)
For all algorithm method except KNN, we categorise the countryRegion, hasVideo, hasGitHub, hasReddit, platform, minInvestment as factor based on its value as below:

 ![image](https://github.com/Pragati2693/ML_Project/assets/68961996/2f60d488-bfc4-4e1d-b49c-bdf38e5477af)


#### Standardizing/Normalising the columns

Standardization and Normalization of columns: Rating, PriceUSD, teamSize, coinNum, Distribution% and Duration as per below function
Standardization                                                                         Normalisation
<img width="206" alt="image" src="https://github.com/Pragati2693/ML_Project/assets/68961996/19c6c741-7ddf-4826-b258-682541a088ef">                     <img width="179" alt="image" src="https://github.com/Pragati2693/ML_Project/assets/68961996/4782f0d6-a24f-4cce-98d6-ef7a967c03a5">





## Evaluation of Methods
 
### KNN Method

The accuracy with Knn method was 63% when used normalization function on the columns whereas, accuracy increased to 67.5% when used Scandalization and change the value of CountryRegion as numeric value for the country with success rate 0 to 10% as 1, 20 to 30% as 2 and so on and for country with success rate between 70 to 100% as 10 as below:

![image](https://github.com/Pragati2693/ML_Project/assets/68961996/5abb0c0a-e7c0-4eea-ba1a-79e0fda8a7b7)

 
Maximum accuracy achieved at K=44
**Confusion matrix (Accuracy = 67.4 %)** 

![image](https://github.com/Pragati2693/ML_Project/assets/68961996/808c9abe-5bec-4358-aa3d-8c824edf02b6)

**ROC Curve (AUC = 0.67)**

![image](https://github.com/Pragati2693/ML_Project/assets/68961996/4638633b-bad4-4750-9180-5a1dda81d11c)

                    

### Random Forest

At ntree = 600 and mtry = 4, the accuracy was maximum, 66.3%

![image](https://github.com/Pragati2693/ML_Project/assets/68961996/42dffcfa-55ce-439d-a17d-b455044e979d)

 

**Confusion Matrix (Accuracy = 66.2%)**   

![image](https://github.com/Pragati2693/ML_Project/assets/68961996/ce33462c-eb11-4879-9608-cc45dbf75db6)

**ROC Curve (AUC = 0.66)**

![image](https://github.com/Pragati2693/ML_Project/assets/68961996/9f93bc88-2bb8-47f9-b4c8-0e563b329e62)

            

### Logistic Regression

### With Logistic Regression, Accuracy is 68.3%

 ![image](https://github.com/Pragati2693/ML_Project/assets/68961996/a62624d4-721d-4be8-866a-dcf5c9aacdea)


**Confusion Matrix (Accuracy = 68.3%)**    

![image](https://github.com/Pragati2693/ML_Project/assets/68961996/c8a4aefb-023b-4290-a8ba-93856386ffff)

**ROC Curve (AUC = 0.69)**

![image](https://github.com/Pragati2693/ML_Project/assets/68961996/fa3b952f-ba6d-49cd-b333-6936383bb305)

                


### SVM Model

Applying ksvm methed and using kernel as “vanilladot” accuracy is 68.9% where as when kernel as “radial” and type as ‘C-Classification” was used accuracy was 61%

 ![image](https://github.com/Pragati2693/ML_Project/assets/68961996/78077cba-b827-40ca-bd26-8655a96421d4)

**Confusion Matrix (Accuracy = 68.9%)**                        
              
![image](https://github.com/Pragati2693/ML_Project/assets/68961996/1346b9bc-26a3-4706-93db-a5d9db767b41)

**ROC Curve (AUC = 0.68)**

![image](https://github.com/Pragati2693/ML_Project/assets/68961996/b572dfe8-8dd3-4893-8a9c-4d614ac8f0f4)


### Decision Tree

Using C5.0 method, accuracy is 65.2% 

![image](https://github.com/Pragati2693/ML_Project/assets/68961996/e915ea52-51f7-4dfa-b81f-bfeff23724e0)

**Confusion Matrix (Accuracy = 65.2%)**   

![image](https://github.com/Pragati2693/ML_Project/assets/68961996/d80861ae-06b0-469d-8a29-1ff32b41755d)


**ROC Curve (AUC = 0.663)**
            
![image](https://github.com/Pragati2693/ML_Project/assets/68961996/bbd8c62b-6a55-4fe4-af25-2c5a26068724)


### XGboost

Using XGBoost algorithm and giving prediction to be positive for value greater than 0.5, accuracy is 65.4%.

![image](https://github.com/Pragati2693/ML_Project/assets/68961996/19f2f595-5f3a-4a28-be69-9d4d95a8c3c6)

 

**Confusion Matrix (Accuracy = 65.4%)**

![image](https://github.com/Pragati2693/ML_Project/assets/68961996/1802daac-3fdb-495e-b77d-9ee2b9101a99)


**ROC Curve ( AUC = 0.67)**

![image](https://github.com/Pragati2693/ML_Project/assets/68961996/0a3f9fcf-de79-4727-83b9-066ebc341eb2)

         

### NaïveBayes

The accuracy with NaiveBayes model is lowest, 50.1% and has very high difference between Sensitivity and Specificity.

![image](https://github.com/Pragati2693/ML_Project/assets/68961996/bb85b46f-aaa5-4205-b98e-29c27b496b8a)

 

**Confusion Matrix (Accuracy = 50.1%)**

![image](https://github.com/Pragati2693/ML_Project/assets/68961996/718dee8a-3664-41ef-8098-13c3020ffa3e)

 

## Analysis of the Result & Conclusion

*kSVM method gives the highest accuracy of approx. 69% but the difference between Sensitivity and Specificity is lowest, approx. .2 with Decision tree method.* 
**Success Rate of all the methods**

 ![image](https://github.com/Pragati2693/ML_Project/assets/68961996/45db8e5c-8348-4f7a-b990-1dbbf18bcc9e)


We now know that kSVM method can predict the success of the company on the basis of other attributes values with highest accuracy. Also, it can be seen that the proportion of the true success out of all the success company (Sensitivity) is high (between 70 to 80%) for all the algorithm except Naive Bayes model and the proportion of the company identified as unsuccessful correctly out of all unsuccessful company (Specificity) is very low for all model except Naive Bayes model.
Thus, overall kSVM model will give better accuracy but for only predicting the companies which could be unsuccessful in raising the funds Naïve Bayes model is better and for only predicting the companies which could be successful in raising the funds Decision tree model is better.


Reference
Bansal, S. (2019) An insightful story of crowdfunding projects! Kaggle.com. [Online]. [Accessed 6 April 2023]. Available from: https://www.kaggle.com/code/shivamb/an-insightful-story-of-crowdfunding-project

