import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np
from typing import List, Dict, Tuple, Optional
import os
import re
from tqdm import tqdm # Import tqdm library

# Local model paths
MODEL_PATHS = {
    'Qwen2.5-3B-Instruct': "llm_weights/Qwen/Qwen2.5-3B-Instruct",
    'DeepSeek-R1-Distill-Qwen-1.5B': "llm_weights/DeepSeek-R1-Distill-Qwen-1.5B",
}

class L3Lite:
    def __init__(self, model_names: Optional[List[str]] = None, device: str = "cuda"):
        """
        Initialize the L3Lite evaluator.

        Args:
            model_names: List of model names to use. If None, all available models will be used.
            device: The device to run the models on.
        """
        # Check if CUDA is available
        if "cuda" in device:
            if not torch.cuda.is_available():
                print(f"Warning: Specified device is {device}, but CUDA is not available. Will use CPU.")
                self.device = "cpu"
            else:
                 self.device = device
        else:
             self.device = device # Use the specified non-CUDA device

        self.models = {}
        self.tokenizers = {}

        # If no models are specified, use all available models
        if model_names is None:
            model_names = list(MODEL_PATHS.keys())

        # Load models and tokenizers
        for model_name in model_names:
            if model_name not in MODEL_PATHS:
                print(f"Warning: Unknown model: {model_name}, skipping loading.")
                continue # Skip unknown model

            model_path = MODEL_PATHS[model_name]
            # Check if model path exists
            if not os.path.exists(model_path):
                 print(f"Warning: Model path not found: {model_path}, skipping loading model {model_name}.")
                 continue # Skip model if path does not exist

            print(f"Loading model: {model_name} from {model_path}")

            try:
                self.tokenizers[model_name] = AutoTokenizer.from_pretrained(model_path)

                # Determine model type and load corresponding class
                if model_name in ['flan-t5-small', 'flan-t5-large', 'flan-t5-xl']:
                    self.models[model_name] = AutoModelForSeq2SeqLM.from_pretrained(
                        model_path,
                        torch_dtype=torch.float16,
                        device_map=self.device # Use self.device
                    )
                else: # Assume other models are CausalLM
                    self.models[model_name] = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        torch_dtype=torch.float16,
                        device_map=self.device # Use self.device
                    )

                # Ensure model is loaded to the correct device
                if self.device != 'cpu' and hasattr(self.models[model_name], 'to'):
                     self.models[model_name].to(self.device)
                self.models[model_name].eval() # Set to evaluation mode

            except Exception as e:
                 print(f"Error loading model {model_name}: {e}, skipping this model.")
                 # If loading fails, remove from loaded dictionaries (if partially loaded)
                 self.models.pop(model_name, None)
                 self.tokenizers.pop(model_name, None)
                 continue # Skip loading current model

        # Cache 1/0 token ids for each model
        self.binary_ids = {}
        if not self.models:
             print("Warning: No models were loaded successfully. L3-Lite will not be able to perform evaluation.")

        for model_name, tokenizer in self.tokenizers.items():
             try:
                # Use a leading space to get independent tokens for words
                one_ids = tokenizer.encode(" 1", add_special_tokens=False)
                zero_ids = tokenizer.encode(" 0", add_special_tokens=False)

                # Check if encoding result is empty
                if not one_ids or not zero_ids:
                     print(f"Warning: Model {model_name} could not encode ' 1' or ' 0' as independent tokens. This model may not be usable for L3-Lite.")
                     # If encoding fails, mark the model as unusable or remove it
                     self.models.pop(model_name, None)
                     self.tokenizers.pop(model_name, None)
                     continue # Skip current model

                # Use the last token as the token id for 1/0
                self.binary_ids[model_name] = {
                    "one": one_ids[-1],
                    "zero": zero_ids[-1]
                }

                print(f"Model {model_name} ' 1' token id: {self.binary_ids[model_name]['one']}")
                print(f"Model {model_name} ' 0' token id: {self.binary_ids[model_name]['zero']}")

             except Exception as e:
                 print(f"Error processing token ids for model {model_name}: {e}, skipping this model.")
                 self.models.pop(model_name, None)
                 self.tokenizers.pop(model_name, None)
                 continue


        if not self.models:
             print("Error: No models available for L3-Lite evaluation.")


    def create_prompt(self, qst: str, pred: str, gt: str) -> str:
        """Creates a prompt to evaluate the semantic similarity of two answers."""
        # Basic cleaning of inputs to prevent errors from None or non-string types
        qst_str = str(qst) if qst is not None else ""
        pred_str = str(pred) if pred is not None else ""
        gt_str = str(gt) if gt is not None else ""

        return f"""I'm evaluating for open QA and need your assistance in determining the answers. The questions, predicted answers and ground truths are as follows. Please determine if the following two answers have the same semantic meaning:\n\
Question:{qst_str}\n\
Answer: {pred_str}\n\
Grount truth: {gt_str}\n\
Please use the questions as background information, provide a similarity score between 0.00 and 1.00, where 1.00 means the answers are completely semantically equivalent, and 0.00 means they are completely different. If the answers are similar, related, or have a contain and be contained relationship, provide a decimal score between 0.00 and 1.00 . Answer with only the number, without any explanation. Your answer : """


    def evaluate_single_model(self, model_name: str, qst:str, pred: str, gt: str) -> Tuple[float, float]:
        """
        Evaluate the semantic similarity of predicted and ground truth answers using a single model.

        Returns:
            (score_one, score_zero): Probability scores for 1 (similar) and 0 (dissimilar) (converted to percentage).
        """
        if model_name not in self.models:
             #print(f"Warning: Model {model_name} is not available, cannot evaluate.")
             return 0.0, 0.0 # Return 0 scores if model is not available

        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]
        # Check if binary_ids exist
        if model_name not in self.binary_ids:
             #print(f"Warning: binary_ids for model {model_name} are not available, cannot evaluate.")
             return 0.0, 0.0

        one_id = self.binary_ids[model_name]["one"]
        zero_id = self.binary_ids[model_name]["zero"]

        prompt = self.create_prompt(qst, pred, gt)
        # print(f"prompt: {prompt}")

        try:
            # Ensure input_ids are not empty
            inputs = tokenizer(prompt, return_tensors="pt")
            if inputs.input_ids.shape[1] == 0:
                 print(f"Warning: Model {model_name} could not encode the prompt.")
                 return 0.0, 0.0

            inputs = inputs.to(self.device)

            # Generate at most 20 tokens
            # Set max_new_tokens a bit larger to prevent the model generating extra tokens that affect number extraction
            # Could also consider setting num_beams=1 and do_sample=False for greedy search
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=20,
                    return_dict_in_generate=True,
                    output_scores=True, # Need output_scores to calculate probabilities
                    pad_token_id=tokenizer.eos_token_id,
                    num_beams=1, # Typically used for generation, L3-Lite's original logic might require this
                    do_sample=False, # Disable sampling, use Greedy Search or Beam Search
                    generation_config=model.generation_config if hasattr(model, 'generation_config') else None # Use model's default generation config
                )

            # Get generated token ids and corresponding scores (log probability)
            # scores is a tuple of tensors, scores[i] are the scores for the i-th generated token
            # generated_ids is the sequence of generated token ids
            generated_ids = outputs.sequences[0, inputs.input_ids.shape[1]:]
            # token_scores are the logits when generating each token
            # Convert logits to probabilities and take the probability of the token at the first time step
            # (usually the model's first generated token is 0 or 1)
            if not outputs.scores:
                 # print(f"Warning: Model {model_name} did not return scores.")
                 return 0.0, 0.0 # Cannot calculate probability if no scores

            # Theoretically, L3-Lite's logic is to look at the probability of the first generated token
            first_token_logits = outputs.scores[0][0] # Logits for the first generated token, batch size = 1
            first_token_probs = torch.softmax(first_token_logits, dim=-1)

            p_one = first_token_probs[one_id].item() if one_id < first_token_probs.size(0) else 0.0 # Check boundary
            p_zero = first_token_probs[zero_id].item() if zero_id < first_token_probs.size(0) else 0.0 # Check boundary

            # If the sum of probabilities for 1 and 0 is very small, the model might have generated other starting tokens
            # In this case, fall back to trying to parse numbers from the generated text
            if p_one + p_zero < 1e-3: # Set a threshold to determine if it's a valid 0/1 start
                # print(f"Warning: Model {model_name} did not generate ' 1' or ' 0' as the first token, trying to parse number.")
                generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                match = re.search(r'(\d+\.?\d*)', generated_text.strip())
                if match:
                    score = float(match.group(1))
                    score = max(0.0, min(1.0, score)) # Ensure score is between 0-1
                    return score * 100, (1.0 - score) * 100 # Convert to percentage
                else:
                     # print(f"Warning: No number found in model {model_name}'s generated text, and it didn't start with 0/1.")
                     return 0.0, 0.0 # Cannot parse, return 0 score

            # Normalize probabilities of 1 and 0 (if they are the first token)
            total_prob = p_one + p_zero
            if total_prob > 0:
                 score_one = p_one / total_prob
                 score_zero = p_zero / total_prob
            else: # Both p_one and p_zero are zero
                 score_one = 0.0
                 score_zero = 0.0 # This case is unlikely if the previous p_one + p_zero < 1e-3 check didn't catch it.

            return score_one * 100, score_zero * 100 # Convert to percentage


        except Exception as e:
            # print(f"Warning: Error evaluating sample with model {model_name}: {e}")
            return 0.0, 0.0 # Return 0 score on error


    def evaluate(self, qst: List[str], preds: List[str], gts: List[str]) -> List[float]:
        """
        Evaluate the semantic similarity between predicted answers and ground truth answers.

        Args:
            qst: List of questions.
            preds: List of predicted answers.
            gts: List of ground truth answers.

        Returns:
            List of L3-Lite scores (percentage, 0-100).
        """
        if not (len(preds) == len(gts) == len(qst)): # More concise check
            print(f"Error: Mismatch in the number of questions, predictions, and ground truths ({len(qst)}, {len(preds)}, {len(gts)}).")
            # Return a list of 0 scores with the same length as predictions, or raise an error
            # Here, choosing to return a list of 0 scores to maintain code flow
            return [0.0] * len(preds) if preds else []


        scores = []

        # Wrap zip iterator with tqdm to display a progress bar
        # No need for enumerate i, tqdm handles progress display
        for qst_item, pred_item, gt_item in tqdm(zip(qst, preds, gts), total=len(preds), desc="Evaluating samples"):

            model_scores_one = []
            # model_scores_zero = [] # Actually only need 'one' scores for L3-Lite calculation

            # Evaluate with each successfully loaded model
            # Check if self.models is empty to avoid iteration when no models are available
            if not self.models:
                 #print("Warning: No models available for L3-Lite evaluation, returning 0 score for this sample.")
                 scores.append(0.0) # No available models, current sample gets 0 score
                 continue # Skip to the next sample

            for model_name in self.models: # Iterate over successfully loaded models in self.models
                score_one, score_zero = self.evaluate_single_model(model_name, qst_item, pred_item, gt_item)
                model_scores_one.append(score_one)
                # model_scores_zero.append(score_zero)


            # Calculate average score (only for models that actually returned scores)
            if model_scores_one: # If any model returned a score
                avg_score_one = np.mean(model_scores_one)
                # avg_score_zero = np.mean(model_scores_zero)
            else: # All models failed to return a score or no models were loaded
                avg_score_one = 0.0


            # L3-Lite score is the average '1' score (indicating similarity), rounded to two decimal places
            l3_lite_score = round(avg_score_one, 2) # Score is already 0-100
            scores.append(l3_lite_score)

            # print(f"  L3-Lite Score for sample: {l3_lite_score:.2f}") # Can be printed occasionally during progress bar updates or removed

        return scores
