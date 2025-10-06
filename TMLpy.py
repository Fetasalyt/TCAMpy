from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from tqdm import tqdm
import TCAMpy as tcam
import pandas as pd
import numpy as np
import random

class TML:
    """
    Class for handling Machine Learning tasks related to the tumor model.
    Allows dataset generation, parameter exploration, and result export.
    Allows predicting the size/confluence for a new set of parameters.
    
    Parameters:
        model (TModel): a created instance of the TModel class.
    """

    def __init__(self, model):
        self.model = model

        self.default_params = {
            "cycles": self.model.cycles,
            "side":   self.model.side,
            "pmax":   self.model.pmax,
            "PA":     self.model.PA,
            "CCT":    self.model.CCT,
            "Dt":     self.model.Dt,
            "PS":     self.model.PS,
            "mu":     self.model.mu,
            "I":      self.model.I,
            "M":      self.model.M,
        }

    def generate_dataset(
            self, n=50, random_params=None,
            output_file="tumor_dataset.csv"
        ):
        """
        Generate a dataset of tumor simulations by randomizing given parameters.

        Parameters:
            n_sims (int): Number of simulations to run.
            randomize_params (dict): Parameters to randomize, e.g.
                {
                    "PA": (1, 20), "PS": (10, 40))
                }
            output_file (str): CSV filename to save dataset.

        Returns:
            pd.DataFrame: Combined DataFrame with all simulation results.
        """

        stats = []

        # Randomize chosen parameters
        for i in tqdm(range(n), desc="Generating simulations"):
            params = self.default_params.copy()
            for key, (low, high) in random_params.items():
                if type(params[key]) == int:
                    params[key] = random.randint(low, high)
                else:
                    params[key] = random.uniform(low, high)

            # Run simulation
            model = tcam.TModel(**params)
            model.run_model(plot = False, animate = False, stats = False)

            run_stats = {}
            for k, v in params.items():
                run_stats[k] = v
            run_stats["Tumor size"] = np.count_nonzero(model.field)
            stats.append(run_stats)

        if stats:
            df = pd.DataFrame(stats)
            df.to_csv(output_file, index=False)
            print(f"Dataset saved to {output_file} ({len(df)} runs)")
            
    def train_predictor(
            self, file, target, test_size=0.2, 
            random_state=42, n_estimators=200
            ):
        """
        Trains a regression model to predict final tumor size based on simulation parameters.
    
        Parameters:
            file (str): CSV file containing the dataset
            target (str): Column name of the target attribute
            test_size (float): Fraction of dataset to use for testing
            random_state (int): Random seed for reproducibility
            n_estimators (int): Number of trees in the random forest
    
        Returns:
            model (RandomForestRegressor): Trained model
            metrics (dict): R^2 and MAE metrics on test set
        """
    
        df = pd.read_csv(file)
        x  = df[df.columns[0:10]]
        y  = df[target]
    
        self.feature_columns = x.columns.tolist()

        # Split into train/test
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, random_state=random_state
        )

        # Train model
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
        model.fit(x_train, y_train)
    
        # Predict & evaluate
        y_pred = model.predict(x_test)
        metrics = {
            "R2": r2_score(y_test, y_pred),
            "MAE": mean_absolute_error(y_test, y_pred)
        }
        self.trained_model = model
    
        print(f"Model trained on {len(x_train)} samples, tested on {len(x_test)}")
        print(f"R^2: {metrics['R2']:.3f}, MAE: {metrics['MAE']:.3f}")
    
        # Feature importance summary
        importance = pd.Series(model.feature_importances_, index=x.columns).sort_values(ascending=False)
        print("\n Top influencing parameters:")
        print(importance.head())
    
        return model, metrics
    
    def predict_new(self, params):
        """
        Predicts an attribute value for a set of
        parameters using a previously trained model.

        Parameters:
            params (list): List of parameters, e.g.
                [500, 50, 10, 1, 24, 1/24, 15, 4, 4, 10]

        Returns:
            float: Predicted tumor size
        """
        if self.trained_model is None:
            raise RuntimeError(
                "No trained model found. Train one with train_predictor() first."
                )
        if self.feature_columns is None:
            raise RuntimeError(
                "Feature column list not found. Did you train the model?"
                )

        if len(params) != len(self.feature_columns):
            raise ValueError(
                f"Parameter list must have {len(self.feature_columns)} values "
                f"(got {len(params)}). Expected order: {self.feature_columns}"
            )
        df = pd.DataFrame([params], columns=self.feature_columns)

        # Ensure all expected features are present (fill missing ones with 0)
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[self.feature_columns]

        return self.trained_model.predict(df)[0]
    

# -- Using the TML Class --

M  = tcam.TModel(500, 50, 10, 1, 24, 1/24, 15, 4, 4, 10)
ml = TML(M)

randomize = {"PA": (1, 15), "M": (0, 10), "I": (0, 10)}

df = ml.generate_dataset(
    n=50,
    random_params=randomize,
    output_file="tumor_dataset.csv"
)

model, metrics = ml.train_predictor("tumor_dataset.csv", "Tumor size")

new_params = [500, 50, 10, 1, 24, 1/24, 15, 4, 4, 10]
print ("Predicted Attribute: ", ml.predict_new(new_params))