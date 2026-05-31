import TCAMpy as tcam

M = tcam.TModel(480, 75, 20, 1, 24, 1/24, 25, 4, 0, 3, 5)
ml = tcam.TML(M)

# Select parameters to randomize and ranges
randomize = {
    "pmax": (15, 20),
    "PA":   (1, 3),
    "CCT":  (12, 48),
    "PS":   (10, 75),
    "mu":   (2, 10),
    "ad":   (0, 10),
    "M":    (0, 10),
    "I":    (0, 5),
    }

# Generate dataset with randomized parameters
df = ml.generate_dataset(
    n=50,
    random_params=randomize,
    output_file="tumor_dataset.csv"
)

# Train a model
model, metrics = ml.train_predictor("tumor_dataset.csv", "Tumor size")

new_params = [480, 75, 15, 1, 24, 1/24, 40, 4, 5, 3, 10]
print ("Predicted Attribute: ", ml.predict_new(new_params))
