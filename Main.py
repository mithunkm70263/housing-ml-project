import os
import pandas as pd
from scipy import stats
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.compose import TransformedTargetRegressor, ColumnTransformer, make_column_selector,make_column_transformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import OneHotEncoder , StandardScaler, FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel

class ClusterSimilarity(BaseEstimator, TransformerMixin):
    
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(
            self.n_clusters,
            random_state=self.random_state
        )
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self

    def transform(self, X):
        return rbf_kernel(
            X,
            self.kmeans_.cluster_centers_,
            gamma=self.gamma
        )

    def get_feature_names_out(self, names=None):
        return [
            f"Cluster {i} similarity"
            for i in range(self.n_clusters)
        ]




DTASETS_DIR = os.path.join("datasets","housing")
CSV_FILE_NAME = "housing.csv"
def load_housing_data(data_dir = DTASETS_DIR,csv_file = CSV_FILE_NAME ):
    csv_path = os.path.join(data_dir,csv_file)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"csv file not found")
    housing_df = pd.read_csv(csv_path)
    return housing_df
housing_df = load_housing_data()

# Create income category
housing_df["income_cat"] = pd.cut(housing_df["median_income"],
                                  bins=[0,1.5,3.0,4.5,6.,np.inf],labels=[1,2,3,4,5])

# Stratified split
strat_train_set,strat_test_set = train_test_split(housing_df,test_size=0.2,
                                                  stratify=housing_df["income_cat"],random_state=42)
# Remove helper column
for set_ in (strat_train_set,strat_test_set):
    set_.drop("income_cat",axis=1,inplace = True)

# Separate features and labels
housing = strat_train_set.drop("median_house_value",axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

housing_num = housing.select_dtypes(include=[np.number])

# Categorical preprocessing
housing_cat = housing[["ocean_proximity"]]
cat_encoder = OneHotEncoder(handle_unknown="ignore",sparse_output=False)
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)

# Test unknown category example
df_test_unknown = pd.DataFrame({
    "ocean_proximity": ["<2H OCEAN", "ISLAND"]
})


encoded_unknown = cat_encoder.transform(df_test_unknown)
df_output = pd.DataFrame(np.asarray(encoded_unknown),
                         columns=cat_encoder.get_feature_names_out(),
                         index=df_test_unknown.index)


model = TransformedTargetRegressor(LinearRegression(),transformer=StandardScaler())
model.fit(housing[["median_income"]],housing_labels)
some_new_data = housing[["median_income"]].iloc[:5]
prediction = model.predict(some_new_data)

#log transform
log_transformer = FunctionTransformer(np.log,inverse_func=np.exp)
log_pop = log_transformer.transform(housing[["population"]])

#similarity to age 35
rbf_transformer = FunctionTransformer(lambda X: rbf_kernel(X, np.array([[35.]]), gamma=0.1))
age_simil_35 = rbf_transformer.transform(housing[["housing_median_age"]])

#similar to san fransisco
sf_coords = (37.7749,-122.41)
sf_transformer = FunctionTransformer(lambda X: rbf_kernel(X, np.array([sf_coords]), gamma=0.1))
sf_simili = sf_transformer.transform(housing[["latitude","longitude"]])

ratio_transformer = FunctionTransformer(lambda X: X[:,[0]]/ X[:,[1]])
ratio_example  = ratio_transformer.transform(np.array([[1.,2.],[3.,4.]]))

#pipeline
num_pipeline = Pipeline([
("impute", SimpleImputer(strategy="median")),
("standardize", StandardScaler()),
])

housing_num_prepared = num_pipeline.fit_transform(housing_num)
df_housing_num_prepared = pd.DataFrame(
    housing_num_prepared, columns=num_pipeline.get_feature_names_out(),
    index=housing_num.index)

#coloumn transformer 
num_attribs = ["longitude","latitude","housing_median_age","total_rooms","total_bedrooms",
               "population","households","median_income"]
cat_attribs = ["ocean_proximity"]
cat_pipeline = make_pipeline(SimpleImputer(strategy="most_frequent"),
                             OneHotEncoder(handle_unknown="ignore"))
def column_ratio(X):
    return X[:, [0]] / X[:, [1]]
def ratio_name(function_transformer, feature_names_in):
    return ["ratio"] # feature names out
def ratio_pipeline():
    return make_pipeline(
SimpleImputer(strategy="median"),
FunctionTransformer(column_ratio, feature_names_out=ratio_name),
StandardScaler())
log_pipeline = make_pipeline(
SimpleImputer(strategy="median"),
FunctionTransformer(np.log, feature_names_out="one-to-one"),
StandardScaler())
cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)
default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"),
StandardScaler())
preprocessing = ColumnTransformer([
("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
("people_per_house", ratio_pipeline(), ["population", "households"]),
("log", log_pipeline, ["total_bedrooms", "total_rooms", "population",
"households", "median_income"]),
("geo", cluster_simil, ["latitude", "longitude"]),
("cat", cat_pipeline, make_column_selector(dtype_include="object")),
],
remainder=default_num_pipeline) # one column remaining: housing_median_age
housing_prepared = preprocessing.fit_transform(housing)

forest_reg = make_pipeline(preprocessing,RandomForestRegressor(random_state=42))
forest_rmses = cross_val_score(forest_reg,housing,housing_labels,
                               scoring="neg_root_mean_squared_error", cv=10)
forest_rmses_scores = -forest_rmses


#randomized search
full_pipeline = Pipeline([
("preprocessing", preprocessing),
("random_forest", RandomForestRegressor(random_state=42,n_jobs=-1)),
])
param_distribs = {"preprocessing__geo__n_clusters": randint(3,50),
                  "random_forest__max_features": randint(2,20)}
rnd_search  = RandomizedSearchCV(full_pipeline,param_distributions=param_distribs,
                                 n_iter=10,cv=3,scoring="neg_root_mean_squared_error",random_state=42)
rnd_search.fit(housing,housing_labels)
final_model = rnd_search.best_estimator_
feature_importances = final_model.named_steps["random_forest"].feature_importances_  # type: ignore

x_test = strat_test_set.drop("median_house_value",axis=1)
y_test = strat_test_set["median_house_value"].copy()
final_predictions = final_model.predict(x_test)  # type: ignore
final_rmse = root_mean_squared_error(y_test,final_predictions)


def rmse(squared_errors):
    return np.sqrt(np.mean(squared_errors))
confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
boot_result = stats.bootstrap([squared_errors], rmse,
confidence_level=confidence, random_state=42)
rmse_lower, rmse_upper = boot_result.confidence_interval