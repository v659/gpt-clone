# chat_inference.py
import argparse
import torch
from pathlib import Path
from typing import List

# import your InferenceEngine class (assumes it's in main.py)
# if your file is named differently, change the import accordingly
from main import InferenceEngine

class ChatSession:
    """
    Thin wrapper around your InferenceEngine that:
     - manages a chat-style prompt ("User: ...\nAssistant:")
     - ensures context length does not exceed model_config.max_sequence_length
    """
    def __init__(self, engine: InferenceEngine, history_keep: int = 1):
        self.engine = engine
        # how many previous turns to keep (simple heuristic)
        self.history = []  # list[str] of user/assistant turns
        self.history_keep = history_keep
        self.max_len = engine.model_config.max_sequence_length

    def _build_prompt(self, user_input: str) -> str:
        # keep last N rounds + current user_input
        rounds = self.history[-(self.history_keep * 2):]  # user/assistant pairs
        # rounds format: ["User: ...", "Assistant: ...", ...]
        prefix = "\n".join(rounds) + ("\n" if rounds else "")
        formatted_prompt = f"{prefix}User: {user_input}\nAssistant:"
        return formatted_prompt

    def _truncate_tokens_if_needed(self, tokens: List[int]) -> List[int]:
        # Ensure we don't exceed position embeddings length.
        # If tokens are longer than max_len, keep the last max_len tokens.
        if len(tokens) > self.max_len:
            return tokens[-self.max_len:]
        return tokens

    def ask(self,
            user_input: str,
            max_new_tokens: int = 50,  # Reduced default
            temperature: float = 0.3,  # Lower default
            top_k: int = 20,  # Lower default
            top_p: float = 0.85,  # Lower default
            repetition_penalty: float = 1.2) -> str:
        prompt = self._build_prompt(user_input)

        response = self.engine.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty
        ).strip()

        # Save to history
        self.history.append(f"User: {user_input}")
        self.history.append(f"Assistant: {response}")

        return response

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True, help="Path to checkpoint (eg. checkpoints/final_model.pt)")
    p.add_argument("--device", type=str, default=None, help="Optional device override: 'cuda','mps','cpu'")
    p.add_argument("--max_new_tokens", type=int, default=150)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--history_keep", type=int, default=1, help="How many previous turns to keep (pairs)")
    return p.parse_args()

def main():
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    # instantiate engine
    engine = InferenceEngine(model_path=str(model_path), device=args.device)
    session = ChatSession(engine, history_keep=args.history_keep)

    print(f"Loaded model ({model_path}) on {engine.device}. Enter 'exit' or 'quit' to stop.")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Bye!")
            break

        resp = session.ask(
            user_input,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p
        )
        print("\nAssistant:", resp, "\n")

if __name__ == "__main__":
    main()
