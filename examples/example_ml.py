import TCAMpy as tcam

M  = tcam.TModel(500, 50, 10, 1, 24, 1/24, 15, 4, 4, 10)
ml = tcam.TML(M)

# Select parameters to randomize and ranges
randomize = {"PA": (1, 15), "M": (0, 10), "I": (0, 10)}

# Generate dataset with randomized parameters
df = ml.generate_dataset(
    n=50,
    random_params=randomize,
    output_file="tumor_dataset.csv"
)

# Train a model
model, metrics = ml.train_predictor("tumor_dataset.csv", "Tumor size")

new_params = [500, 50, 10, 1, 24, 1/24, 15, 4, 4, 10]
print ("Predicted Attribute: ", ml.predict_new(new_params))
