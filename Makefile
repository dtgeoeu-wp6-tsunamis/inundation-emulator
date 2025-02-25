# Makefile

# Declare these targets as phony to indicate they don't produce an actual file
.PHONY: create_emulator run_tensorboard clean predictions

emulator:
	@echo "Creating emulator..."
	poetry run python create_emulator.py 

predictions:
	@echo "Make predictions..."
	poetry run python -m predict

tensorboard:
	@echo "Starting TensorBoard..."
	poetry run tensorboard --logdir generated/logs --port 6006

clean:
	@echo "Cleaning runs (models) and tensorboard logs..."
	rm -rf generated/*
	@echo "Cleanup done."
