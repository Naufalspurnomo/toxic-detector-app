"""
Train model script
"""
from modules.training import train_model

if __name__ == "__main__":
    print("Starting model training...")
    results = train_model()
    
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(f"Accuracy:  {results['metrics']['accuracy']*100:.2f}%")
    print(f"Precision: {results['metrics']['precision']*100:.2f}%")
    print(f"Recall:    {results['metrics']['recall']*100:.2f}%")
    print(f"F1-Score:  {results['metrics']['f1_score']*100:.2f}%")
    print(f"CV Mean:   {results['cross_validation']['mean']*100:.2f}%")
    print("="*50)
