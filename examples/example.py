# Initialize the model from JSON
manager = ModelManager("text_classifier.json")

# Prepare training data
training_data = load_dataset("sentiment_data.csv")  # your data loading function

# Train the model using a specific training configuration
manager.train(
    training_name="train_sentiment",
    data=training_data,
    epochs=10,
    batch_size=32,
    learning_rate=0.001
)

# Save and load
# Save a checkpoint
checkpoint_path = manager.save_checkpoint("sentiment_model_v1")

# Load a checkpoint
manager.load_checkpoint(checkpoint_path)

# List available checkpoints
checkpoints = manager.list_checkpoints()

# Run inference
# Prepare input
input_data = {
    "text": "I really enjoyed this movie, it was fantastic!"
}

# Run inference using a specific inference configuration
result = manager.inference("predict_sentiment", input_data)
print(f"Sentiment: {result['sentiment']}, Confidence: {result['confidence']}")