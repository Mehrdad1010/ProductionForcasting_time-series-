import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class data_prepear:
    
    def __init__(self, df, target_col ="target", invert=False, filter_value=90) :
        self.df = df
        self.target_column =  target_col
        self.invert = invert
        self.filter_value = filter_value


    def make_lags(self, ts, lags, lead_time=1):
        
        return pd.concat(
            {
                f'y_lag_{i}': ts.shift(i)
                for i in range(lead_time, lags + lead_time)
            },
            axis=1)
    

    def make_roling_lag(self ,df, name="QG"):
        return pd.concat(
            [
                df.shift(1).rolling(7).mean().rename(f"{name}_lagged_mean_7D"),
                df.shift(1).rolling(7).max().rename(f"{name}_lagged_max_7D"),
                df.shift(1).rolling(7).min().rename(f"{name}_lagged_min_7D"),
                df.shift(1).rolling(15).mean().rename(f"{name}_lagged_mean_15D"),
                df.shift(1).rolling(15).max().rename(f"{name}_lagged_max_15D"),
                df.shift(1).rolling(15).min().rename(f"{name}_lagged_min_15D"),
                df.shift(1).rolling(30).mean().rename(f"{name}_lagged_mean_30D"),
                df.shift(1).rolling(30).max().rename(f"{name}_lagged_max_30D"),
                df.shift(1).rolling(30).min().rename(f"{name}_lagged_min_30D"),
            ],
            axis="columns"
        )

    def make_multistep_target(self, ts, steps):
        return pd.concat(
            {f'y_step_{i + 1}': ts.shift(-i)
             for i in range(steps)},
            axis=1)
    


    def creator(self):
        self.df = self.df.reset_index().drop("date", axis=1)
        trget_index = self.df[self.df["AVG_CHOKE_SIZE_P"]>self.filter_value].index
        target = self.df[self.target_column]
        feature = self.df.drop(self.target_column, axis=1)

        # instance of the StandardScaler for the feature
        sc_f = StandardScaler()
        # instance of the StandardScaler for the target
        sc_t = StandardScaler()

        # fit the scaler to the feature
        
        feature_scaled = sc_f.fit_transform(feature.values)
        # fit the scaler to the target
        target_scaled = sc_t.fit_transform(target.values.reshape(-1, 1))

        # convert numpy scled to pandas
        feature_scales = pd.DataFrame(feature_scaled, columns=feature.columns)
        target_scales = pd.DataFrame(target_scaled, columns=["QG"])

        X0 = self.make_lags(feature_scales, lags=30)
        X0.columns = [' '.join(col).strip() for col in X0.columns.values]

        X1 = self.make_lags(target_scales, lags=30)
        X1.columns = [" ".join(col).strip() for col in X1.columns.values]

        X = pd.concat(
            [
                feature_scales,
                target_scales,
                X0,
                X1,
                self.make_roling_lag(target_scales["QG"], name="QG"),
                self.make_roling_lag(feature_scales["bhp"], name="bhp"),
                self.make_roling_lag(feature_scales["bht"], name="bht"),
                self.make_roling_lag(feature_scales["dp_tubing"], name="dp_tubing"),
                self.make_roling_lag(feature_scales["AVG_CHOKE_SIZE_P"], name="AVG_CHOKE_SIZE_P")
            ],
            axis="columns",
        ).fillna(0.0)
        

        # Eight-week forecast
        y = self.make_multistep_target(target_scales, steps=10).dropna()
        y.columns = [" ".join(col).strip() for col in y.columns.values]
        # Shifting has created indexes that don't match. Only keep times for
        # which we have both targets and features.
        y, X = y.align(X, join='inner', axis=0)


        # Set the n_components=3
        principal=PCA(n_components=15)
        principal.fit(X)
        X_PCA = principal.transform(X)

        series_X = pd.DataFrame(X_PCA, columns=["1", "2", "3", "4", "5",
        "6", "7", "8", "9", "10",
        "11", "12", "13", "14", "15"])

        X_train = series_X.drop(trget_index)
        y_train = y.drop(trget_index)
        X_test = series_X.iloc[trget_index]
        y_test = y.iloc[trget_index]
        return X_train, y_train, X_test, y_test


    