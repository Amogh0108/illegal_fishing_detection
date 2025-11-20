"""Run complete end-to-end pipeline"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.logger import setup_logger
from src.preprocessing.clean_ais import main as clean_ais
from src.preprocessing.eez_filter import main as filter_eez
from src.features.extract_features import main as extract_features
from src.models.train import main as train_models
from src.evaluation.baseline import main as run_baseline
from src.models.ensemble import main as run_ensemble
from src.evaluation.metrics import main as evaluate

logger = setup_logger(__name__, "logs/pipeline.log")

def run_full_pipeline():
    """Execute complete pipeline"""
    logger.info("=" * 70)
    logger.info("STARTING FULL IUU FISHING DETECTION PIPELINE")
    logger.info("=" * 70)
    
    try:
        # Step 1: Data Preprocessing
        logger.info("\n[STEP 1/7] Cleaning AIS data...")
        clean_ais()
        
        # Step 2: EEZ Filtering
        logger.info("\n[STEP 2/7] Filtering data within EEZ...")
        filter_eez()
        
        # Step 3: Feature Extraction
        logger.info("\n[STEP 3/7] Extracting features...")
        extract_features()
        
        # Step 4: Model Training
        logger.info("\n[STEP 4/7] Training ML models...")
        train_models()
        
        # Step 5: Baseline Detection
        logger.info("\n[STEP 5/7] Running rule-based baseline...")
        run_baseline()
        
        # Step 6: Ensemble Prediction
        logger.info("\n[STEP 6/7] Running ensemble prediction...")
        run_ensemble()
        
        # Step 7: Evaluation
        logger.info("\n[STEP 7/7] Evaluating models...")
        evaluate()
        
        logger.info("\n" + "=" * 70)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)
        logger.info("\nOutputs saved to:")
        logger.info("  - Processed data: data/processed/")
        logger.info("  - Trained models: outputs/models/")
        logger.info("  - Predictions: outputs/anomaly_predictions.csv")
        logger.info("  - Evaluation: outputs/evaluation/")
        logger.info("\nTo launch dashboard, run:")
        logger.info("  python src/dashboard/app.py")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    run_full_pipeline()
