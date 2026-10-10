from models.rf_classifier import train_rf_classifier
from src.evaluator import generate_reports

results = train_rf_classifier()
generate_reports(results)