<p align="center">
<img width="" src="https://media0.giphy.com/media/4FQMuOKR6zQRO/giphy.gif?cid=ecf05e47q5dsu5w71qypmr5phjo3vyckjmkbsybvju1iylkr&rid=giphy.gif&ct=g" align="center" alt="gif" />
<h1 align="center">2022 Python for Machine Learning & Data Science Masterclass 

</h1>
</p>


I am doing the course "2022 Python for Machine Learning & Data Science Masterclass", you can find all projects for each section on this repository.
I really like Jose Portilla course, on this course you will learn the concepts, math, statistics and python coding of the most used machine learning models.
For someone that doesn't know about this course, follow this link:
(https://www.udemy.com/course/python-for-machine-learning-data-science-masterclass/).

<p align="center">
<img width="" src="https://gigacourse.com/wp-content/uploads/2021/08/321545555.jpg" align="center" alt="gif" />
</p>

Summary of what I learned trough this course:
- How to use data science and machine learning with Python.
- Create data pipeline workflows to analyze, visualize, and gain insights from data.
- Build a portfolio of data science projects with real world data.
- Be able to analyze your own data sets and gain insights through data science.
- Master critical data science skills.
- Understand Machine Learning from top to bottom.
- Replicate real-world situations and data reports.
- Use NumPy for numerical processing with Python.
- Conduct feature engineering on real world case studies.
- Use Pandas for data manipulation with Python.
- Create supervised machine learning algorithms to predict classes.
- Use Matplotlib to create fully customized data visualizations with Python.
- Create regression machine learning algorithms for predicting continuous values.
- Use Seaborn to create beautiful statistical plots with Python.
- Construct a modern portfolio of data science and machine learning resume projects.
- Use how to use Scikit-learn to apply powerful machine learning algorithms.
- Get set-up quickly with the Anaconda data science stack environment.
- Use best practices for real-world data sets.
- Understand the full product workflow for the machine learning lifecycle.
- Explore how to deploy your machine learning models as interactive APIs.

## Sections of the Course

✔ Section 1: Introduction to Course<br>
✔ Section 2: OPTIONAL: Python Crash Course<br>
✔ Section 3: Machine Learning Pathway Overview<br>
✔ [Section 4](section_04_numpy): NumPy<br>

> - Numpy Arrays, Random distributions of data, Key attributes and method calls of Numpy arrays, Indexing and Selection, 
Numpy Operators.

✔ [Section 5](section_05_pandas): Pandas<br>

> - Series, Operations, DataFrames, Read csv files, Grabbing information from DataFrame, Columns, Grabbing information based on index and add more rows, Condtion Formating/Filter, Useful Methods (apply,vectorize,replace,nlargest) Describing and Sorting Methods, Identify and remove duplicates, Handle Missing Data, Groupby Operations, Concatenation, Merge, Text Methods, Time Methods, Data Input and Output, SQL (just basics).

✔ [Section 6](section_06_matplotlib): Matplotlib<br>

> - Basics, Figure Object, Figure Parameters, Subplots, Legends, Styling.

✔ [Section 7](section_07_seaborn): Seaborn Data Visualizations<br>

> - Scatterplots, Distribution Plots (Rug Plot, Histogram, KDE Plot), Categorical Plots (Countplot, Barplot, Boxplot, Violinplot, Swarmplot, Boxenplot), Comparison Plots (Jointplot, Pairplot), Seaborn Grids (Catplot, PairGrid), Matrix Plots (Heatmap, Clustermap)

✔ [Section 8](section_08_capstone_project_data_analyst): Data Analysis and Visualization Capstone Project Exercise<br>

> - Scatterplots, Correlation, Countplot, KDE Plot, Regplot, Histoplot, Merge, move_legend function, Clustermap, nsmallest

✔ Section 9: Machine Learning Concepts Overview<br>
✔ [Section 10](section_10_linear_regression): Linear Regression<br>

> Linear Regression
> - Scikit-learn.linear_model (Linear Regression), Scikit-learn.metrics (mean_absolute_error,mean_squared_error), Residuals Plot, Probability Plot, model.coef_, model deployment (joblib) <br>

> Polynomial Regression
> - Scikit-learn.preprocessing (PolynomialFeatures), Scikit-learn.model_selection (train_test_split), Polynomial Complexity x RSME, Scikit-learn.linear_model (Linear Regression), model deployment (joblib) <br>

> Regularization
> - Scikit-learn.preprocessing (PolynomialFeatures, StandardScaler), Scikit-learn.model_selection (train_test_split), Scikit-learn.linear_model (Ridge, RidgeCV, LassoCV, ElasticNetCV), Scikit-learn.metrics (mean_absolute_error,mean_squared_error, SCORERS), hyperparamenters tunning<br>

✔ [Section 11](section_11_feature_engineering): Feature Engineering and Data Preparation<br>

> Dealing with outliers
> - Distplot, Boxplot, Heatmap, corr(), Scatterplot <br>

> Dealing with Missing Data
> - info, describe, isnull().sum(), function and barplot for percent_missing, Feature Extraction, Filling in Data, Drop rows, Imputation of Missing Data, transform, groupby <br>

> Dealing with Categorical Data
> - pd.get_dummies, select_dtypes, pd.concat, corr()<br>


✔ [Section 12](section_12_cross_validation_and_linear_regression_project): Cross Validation, Grid Search, and the Linear Regression Project<br>

**Train | Test Split Procedure**
1. Clean and adjust data as necessary for X and y
2. Split Data in Train/Test for both X and y
3. Fit/Train Scaler on Training X Data
4. Scale X Test Data
5. Create Model
6. Fit/Train Model on X Train Data
7. Evaluate Model on X Test Data (by creating predictions and comparing to Y_test)
8. Adjust Parameters as Necessary and repeat steps 5 and 6

**Train | Validation | Test Split Procedure**
1. Clean and adjust data as necessary for X and y
2. Split Data in Train/Validation/Test for both X and y
3. Fit/Train Scaler on Training X Data
4. Scale X Eval Data
5. Create Model
6. Fit/Train Model on X Train Data
7. Evaluate Model on X Evaluation Data (by creating predictions and comparing to Y_eval)
8. Adjust Parameters as Necessary and repeat steps 5 and 6
9. Get final metrics on Test set (not allowed to go back and adjust after this!)

> Cross Validation
> - Train | Test Split Procedure, Train | Validation | Test Split Procedure, Scikit-learn.model_selection (cross_val_score, cross_validate) <br>

> Grid Search
> - Scikit-learn.model_selection (GridSearchCV), best_estimator_, best_params_, cv_results_ <br>

✔ [Section 13](section_13_logistic_regression): Logistic Regression<br>

> Logistic Regression
> - Scikit-learn.model_selection (LogisticRegression), predict_proba, Scikit-learn.metrics (accuracy_score, confusion_matrix, classification_report, plot_confusion_matrix, precision_recall_curve,plot_precision_recall_curve,plot_roc_curve) <br>

> Multi Class Logistic Regression
> - Scikit-learn.model_selection (LogisticRegression, GridSearchCV), Scikit-learn.metrics (accuracy_score, confusion_matrix, classification_report, plot_confusion_matrix, roc_curve, auc)<br>

> Logistic-Regression-Project-Exercise
> - visualization of the coefficients using a barplot <br>

✔ [Section 14](section_14_K_nearest-neighbors): KNN-KNearest Neighbors<br>

> KNN-Classification
> - Scatterplot parameters, Scikit-learn.neighbors (KNeighborsClassifier), Scikit-learn.pipeline (Pipeline), Scikit-learn.model_selection (GridSearchCV), Scikit-learn.metrics (classification_report, confusion_matrix, accuracy_score), Elbow Method, Full Cross Validation Grid Search for K Value, cv_results_.keys() <br>

> KNN-Exercise
> - Heatmap, map function, mean test scores per K value using .cvresults dictionary <br>

✔ [Section 15](section_15_SVM): Support Vector Machines<br>

> SVM-Classification
> - Scatterplot with hyperplane, Scikit-learn.svm (SVM), svm_margin_plot (plot_svm_boundary) <br>

> SVM-Regression
> - Heatmap parameters, Scikit-learn.svm (SVM, LinearSVR), Scikit-learn.metrics (mean_absolute_error,mean_squared_error), Scikit-learn.model_selection (GridSearchCV) <br>

> SVM-Project-Exercise
> - unbalanced dataset, map function, visualization of the coefficients using a barplot <br>

✔ [Section 16](section_16_decision_trees): Tree Based Methods: Decision Tree Learning<br>

> Decision-Trees
> - pd.get_dummies, Scikit-learn.tree (DecisionTreeClassifier, plot_tree), DataFrame of feature_importances_, report_model function (precision, recall, f1-score and plot_tree) <br>

✔ [Section 17](section_17_random_forests): Random Forests<br>

> Random-Forest-Classification
> - pd.get_dummies, Scikit-learn.ensemble (RandomForestClassifier), hyperparameter tunning <br>

> Random-Forest-Regression
> - sinusoid data set distribution, best models comparison (LinearRegression, PolynomialFeatures, KNeighborsRegressor, DecisionTreeRegressor, SVR, RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor) <br>

✔ [Section 18](section_18_boosting_methods): Boosting Methods<br>

> Adaboosting
> - unique barplot visualization, Scikit-learn.ensemble (AdaBoostClassifier), Scikit-learn.metrics (classification_report,plot_confusion_matrix,accuracy_score), feature_importances_, feature_importances_.argmax(), lineplot of performance x number of features added, feature_importances_ DataFrame and barplot visualization <br>

> Gradient-Boosting
> - pd.get_dummies, Scikit-learn.ensemble (GradientBoostingClassifier), Scikit-learn.model_selection (GridSearchCV), feature_importances_ DataFrame and barplot visualization <br>

✔ [Section 19](section_19_supervised_learning_capstone_project_cohort_analysis): Supervised Learning Capstone Project- Cohort Analysis and Tree Based Methods<br>

> supervised_learning_capstone_project_cohort_analysis
> - Feature correlation barplot visualization, multiple distplot, lineplot of churn rate per months, new feature creation by joining months<br>

✔ [Section 20](section_20_naive_bayes_classification_and_NLP): Naive Bayes Classification and Natural Language Processing (Supervised Learning) <br>

> Feature-Extraction-From-Text
> - Bag of Words, Bag of Words to Frequency Counts, Concepts (Bag of Words and Tf-idf, Stop Words and Word Stems, Tokenization and Tagging), Scikit-learn.feature_extraction.text (TfidfTransformer,TfidfVectorizer,CountVectorizer) <br>

> Text-Classification
> - Scikit-learn.feature_extraction.text (TfidfVectorizer), Scikit-learn.naive_bayes (MultinomialNB), PipeLine for Deployment <br>

✔ Section 21: Unsupervised Learning<br>
✔ [Section 22](section_22_kmeans_clustering): K-Means Clustering<br>

> Kmeans-Clustering
> -  <br>

> Kmeans-Color-Quantization
> -  <br>

> Kmeans-Clustering-Project-Exercise
> -  <br>

✔ [Section 23](section_23_hierarchical_clustering): Hierarchical Clustering<br>

> Teste

✔ [Section 24](section_24_DBSCAN): DBSCAN-Density-based spatial clustering of applications with noise<br>

> Teste

✔ [Section 25](section_25_PCA): PCA- Principal Component Analysis and Manifold Learning<br>

> Teste

✔ Section 26: Model Deployment<br>
[DATA](DATA) - All data used throughout the course

## 🛠 Tools and technologies

- Python 3
- Jupyter Notebook
- Python Scripting and Automation
- Data Science
- Pandas
- NumPy
- Matplotlib
- Plotly
- Scikit learn
- Seaborn
- Git, GitHub and Version Control
- APIs
- Databases
- Model deployment


<p align="center">
<img width="" src="https://udemy-certificate.s3.amazonaws.com/image/UC-f1d333c3-8043-4df3-8747-aaa090c9d456.jpg?v=1663366755000" align="center" alt="gif" />
</p>

This certificate above verifies that Bruno Aguiar successfully completed the course 2022 Python for Machine Learning & Data Science Masterclass on 09/16/2022 as taught by Jose Portilla on Udemy. The certificate indicates the entire course was completed as validated by the student. The course duration represents the total video hours of the course at time of most recent completion.

<hr>

Bruno Aguiar September 16, 2022
