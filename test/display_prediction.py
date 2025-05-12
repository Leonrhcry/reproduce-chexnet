import visualize_prediction as V
import pandas as pd
import warnings

# Suppress PyTorch warnings about source code changes
warnings.filterwarnings('ignore')

# Variables
STARTER_IMAGES = True
PATH_TO_IMAGES = "starter_images"
PATH_TO_MODEL = 'pretrained/checkpoint'
LABEL = "Pneumonia"
POSITIVE_FINDINGS_ONLY = True

# Wrap the code inside the main block to handle process spawning safely
if __name__ == '__main__':
    # Load data and model
    dataloader, model = V.load_data(PATH_TO_IMAGES, LABEL, PATH_TO_MODEL, POSITIVE_FINDINGS_ONLY, STARTER_IMAGES)
    
    # Print number of cases for review
    print("Cases for review:")
    print(len(dataloader))
    
    # Get predictions
    preds = V.show_next(dataloader, model, LABEL)
    
    # Return the predictions
    preds
