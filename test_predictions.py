"""
Test predictions script
"""
from modules.predictor import ToxicityPredictor

if __name__ == "__main__":
    p = ToxicityPredictor()
    
    print("=== TEST PREDICTIONS ===\n")
    
    tests = [
        ("dasar nubs bego roblox", "Toxic"),
        ("bocil toxic bacot mulu", "Toxic"),
        ("anjing lo main kayak sampah", "Toxic"),
        ("tolol banget sih main game", "Toxic"),
        ("main bareng yuk seru", "Safe"),
        ("gg wp mantap gamenya", "Safe"),
        ("keren banget skillnya bro", "Safe"),
        ("makasih udah carry tim", "Safe"),
    ]
    
    correct = 0
    for text, expected in tests:
        result = p.predict(text)
        is_correct = (result['label_name'] == "Aman" and expected == "Safe") or \
                     (result['label_name'] == "Toxic" and expected == "Toxic")
        correct += 1 if is_correct else 0
        status = "✓" if is_correct else "✗"
        print(f"{status} '{text}'")
        print(f"   → {result['label_name']} ({result['confidence']*100:.0f}%) [Expected: {expected}]")
        print()
    
    print(f"Accuracy: {correct}/{len(tests)} = {correct/len(tests)*100:.0f}%")
