import argparse
import json
from L3_Lite import L3Lite

def main():
    parser = argparse.ArgumentParser(description="Evaluate prediction results using L3-Lite")
    parser.add_argument("--model_names", nargs="+", default=['Qwen2.5-3B-Instruct'], help="List of model names to use")
    parser.add_argument("--device", type=str, default='cuda:3', help="Device to run on")
    parser.add_argument("--result_path", type=str, default='/data/zhangyu/tmp/results/result.json', help="Path to the results file")
    args = parser.parse_args()

    # Initialize the L3-Lite evaluator
    l3_lite = L3Lite(model_names=args.model_names, device=args.device)

    # Read the results file
    with open(args.result_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    # Extract questions, predictions, and ground truths
    questions = [item['question'] for item in results]
    predictions = [item['pred'] for item in results]
    ground_truths = [item['gt'] for item in results]

    # Run evaluation
    scores = l3_lite.evaluate(questions, predictions, ground_truths)

    # Print results
    print("\nEvaluation Results:")
    total_score = 0
    for i, (result, score) in enumerate(zip(results, scores)):
        print(f"\nSample {i+1}:")
        print(f"Image: {result['image']}")
        print(f"Question Type: {result['question_type']}")
        print(f"Question: {result['question']}")
        print(f"Prediction: {result['pred']}")
        print(f"Ground Truth: {result['gt']}")
        print(f"L3-Lite Score: {score:.4f}")
        print(f"Explanation: {'Semantically Similar' if score > 0.5 else 'Semantically Different'}") # 0-1 scale from L3-Lite
        total_score += score

    # Print average score
    if scores: # Avoid division by zero if scores list is empty
        avg_score = total_score / len(scores)
        print(f"\nOverall Evaluation Results:")
        print(f"Number of Samples: {len(scores)}")
        print(f"Average L3-Lite Score: {avg_score:.4f}")
    else:
        print("\nNo samples were evaluated.")


if __name__ == "__main__":
    main()
