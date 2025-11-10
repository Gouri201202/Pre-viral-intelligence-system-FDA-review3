import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")

# Validate API keys
if not YOUTUBE_API_KEY:
    print("⚠️  WARNING: YOUTUBE_API_KEY not found in .env file")

# Model configurations
SPAM_MODEL_NAME = "unitary/toxic-bert"  # Alternative: "martin-ha/toxic-comment-model"
SENTIMENT_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
LLM_MODEL_NAME = "microsoft/DialoGPT-medium"

# File paths
DATA_DIR = "data"
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(DATA_DIR, "models")
LOGS_DIR = "logs"

# Create directories if they don't exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# API settings
YOUTUBE_API_QUOTA_PER_DAY = 10000
MAX_COMMENTS_PER_REQUEST = 100

# Model settings
MAX_TEXT_LENGTH = 512
BATCH_SIZE = 16
DEVICE = "cpu"  # Will auto-detect GPU if available

# Viral prediction thresholds
VIRAL_THRESHOLD_HIGH = 0.7
VIRAL_THRESHOLD_MEDIUM = 0.4
SPAM_THRESHOLD = 0.5

print("✅ Configuration loaded successfully!")
