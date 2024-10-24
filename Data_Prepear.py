import pandas as pd
from sklearn.preprocessing import StandardScaler

class data_prepear:
    
    def __init__(self, df, target_col ="target", invert=False) :
        self.df = df
        self.target_column =  target_col
        self.invert = invert


    def make_lags(self,ts, lags, lead_time=1):
        
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

        target = self.df[self.target_column]
        feature = self.df.drop(self.target_column, axis=1)

        # instance of the StandardScaler for the feature
        sc_f = StandardScaler()
        # instance of the StandardScaler for the target
        sc_t = StandardScaler()

        # fit the scaler to the feature
        feature_scaled = sc_f.fit_transform(feature)
        # fit the scaler to the target
        target_scaled = sc_t.fit_transform(target)

        # convert numpy scled to pandas
        feature_scales = pd.DataFrame(feature_scaled, columns=sc_f.columns)
        target_scales = pd.DataFrame(target_scaled, columns=sc_t.columns)

        X0 = self.make_lags(feature_scales, lags=30)
        X0.columns = [' '.join(col).strip() for col in X0.columns.values]

        X1 = self.mak_lags(target_scales, lags=30)
        X1.columns = [" ".join(col).strip() for col in X1.columns.values]

        X = pd.concat(
            [
                feature_scales,
                X0,
                X1,
                self.make_roling_lag(target_scales, name="QG"),
                self.make_roling_lag(feature_scales["bhp"], name="bhp"),
                self.make_roling_lag(feature_scales["bht"], name="bht"),
                self.make_roling_lag(feature_scales["dp_tubing"], name="dp_tubing"),
                self.make_roling_lag(feature_scales["AVG_CHOKE_SIZE_P"], name="AVG_CHOKE_SIZE_P")
            ],
            axis="columns",
        ).fillna(0.0)
        

        # Eight-week forecast
        y = self.make_multistep_target(df["QG"], steps=10).dropna()

        # Shifting has created indexes that don't match. Only keep times for
        # which we have both targets and features.
        y, X = y.align(X, join='inner', axis=0)




    