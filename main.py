import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import tiktoken
from datasets import load_dataset
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import logging
import wandb
import numpy as np
from dataclasses import dataclass, asdict
from contextlib import nullcontext
import os

os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

# ----------------------------
# Setup Logging
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ----------------------------
# Constants
# ----------------------------
class TextFormat:
    """Centralized text formatting constants"""
    USER_PREFIX = "User: "
    ASSISTANT_PREFIX = "Assistant: "
    EOS_TOKEN = "<|endoftext|>"

    @staticmethod
    def format_conversation(prompt: str, response: str) -> str:
        """Format a conversation pair"""
        return f"{TextFormat.USER_PREFIX}{prompt}\n{TextFormat.ASSISTANT_PREFIX}{response}\n{TextFormat.EOS_TOKEN}"

    @staticmethod
    def format_prompt(prompt: str) -> str:
        """Format a prompt for inference"""
        return f"{TextFormat.USER_PREFIX}{prompt}\n{TextFormat.ASSISTANT_PREFIX}"


class GenerationConfig:
    """Generation hyperparameters"""
    REPETITION_PENALTY_WINDOW = 20
    REPETITION_PENALTY_DECAY = 2.0
    REPETITION_STOP_THRESHOLD = 10
    REPETITION_PATTERN_LENGTH = 5
    MAX_SAME_TOKEN_COUNT = 10


def collate_with_pad(batch, pad_token_id=50256):
    """Wrapper for collate_fn"""
    return collate_fn(batch, pad_token_id=pad_token_id)


# ----------------------------
# Configuration
# ----------------------------
@dataclass
class ModelConfig:
    """Model architecture configuration"""
    vocab_size: int = 50257
    embed_dim: int = 768
    num_heads: int = 12
    num_layers: int = 12
    ff_dim: int = 3072
    max_sequence_length: int = 1024
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    initializer_range: float = 0.02

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        """Validate that embed_dim is divisible by num_heads"""
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) must be divisible by num_heads ({self.num_heads}). "
                f"Head dimension would be {self.embed_dim / self.num_heads}"
            )


@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    batch_size: int = 32
    gradient_accumulation_steps: int = 2
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    num_epochs: int = 10
    warmup_steps: int = 500
    max_grad_norm: float = 1.0
    eval_steps: int = 250
    save_steps: int = 1000
    max_samples_per_dataset: Optional[int] = None
    seed: int = 42
    use_fp16: bool = True
    use_ddp: bool = False
    num_workers: int = 4
    sample_generation_steps: int = 100
    nan_check_interval: int = 50  # Check for NaN every N steps

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ----------------------------
# Tokenizer with Caching
# ----------------------------
class ProductionTokenizer:
    """Enhanced tokenizer with caching and special tokens"""

    def __init__(self, encoding_name: str = "gpt2"):
        self.enc = tiktoken.get_encoding(encoding_name)
        self.pad_token_id = 50256
        self.eos_token_id = 50256
        self.bos_token_id = 50256
        self.vocab_size = self.enc.n_vocab
        self._cache = {}

    def encode(self, text: str, max_length: Optional[int] = None) -> List[int]:
        """Encode text with optional caching"""
        if text in self._cache:
            tokens = self._cache[text]
        else:
            tokens = self.enc.encode(text, allowed_special={"<|endoftext|>"})
            self._cache[text] = tokens

        if max_length:
            tokens = tokens[:max_length]
        return tokens

    def decode(self, tokens: List[int]) -> str:
        """Decode tokens to text"""
        valid_tokens = [t for t in tokens if t < self.vocab_size]
        return self.enc.decode(valid_tokens)

    def clear_cache(self):
        """Clear encoding cache"""
        self._cache.clear()


# ----------------------------
# Dataset
# ----------------------------
class ProductionConversationDataset(Dataset):
    """High-quality conversation dataset with intelligent filtering"""

    def __init__(self,
                 dataset_name: str = "HuggingFaceH4/ultrachat_200k",
                 split: str = "train_sft",
                 tokenizer: ProductionTokenizer = None,
                 max_length: int = 1024,
                 max_samples: Optional[int] = None,
                 cache_dir: Optional[str] = None):
        self.tokenizer = tokenizer or ProductionTokenizer()
        self.max_length = max_length
        self.cache_file = self._get_cache_path(dataset_name, split, max_length, cache_dir)

        if self.cache_file and self.cache_file.exists():
            logger.info(f"Loading cached dataset from {self.cache_file}")
            cache_data = torch.load(self.cache_file)
            self.samples = cache_data['samples']
            logger.info(f"Loaded {len(self.samples)} cached samples")
            return

        logger.info(f"Loading dataset {dataset_name}...")
        dataset = self._load_dataset(dataset_name, split, max_samples)

        logger.info("Processing conversations...")
        self.samples = self._process_conversations(dataset)
        logger.info(f"Processed {len(self.samples)} samples")

        if self.cache_file:
            torch.save({'samples': self.samples}, self.cache_file)
            logger.info(f"Cached dataset to {self.cache_file}")

    @staticmethod
    def _get_cache_path(dataset_name: str, split: str, max_length: int,
                        cache_dir: Optional[str]) -> Optional[Path]:
        """Generate a consistent cache file path"""
        if not cache_dir:
            return None
        Path(cache_dir).mkdir(exist_ok=True)
        return Path(cache_dir) / f"{dataset_name.replace('/', '_')}_{split}_{max_length}_cache.pt"

    @staticmethod
    def _load_dataset(dataset_name: str, split: str, max_samples: Optional[int]):
        """Load dataset with fallback to streaming"""
        try:
            dataset = load_dataset(dataset_name, split=split)
            if max_samples:
                dataset = dataset.select(range(min(max_samples, len(dataset))))
        except Exception as e:
            logger.warning(f"Failed to load dataset: {e}. Trying streaming mode...")
            dataset = load_dataset(dataset_name, split=split, streaming=True)
            if max_samples:
                dataset = dataset.take(max_samples)
        return dataset

    def _process_conversations(self, dataset) -> List[List[int]]:
        """Process and filter conversations with quality checks"""
        samples = []
        seen = set()

        for item in tqdm(dataset, desc="Processing conversations"):
            try:
                messages = item.get('messages', [])
                if len(messages) < 2:
                    continue

                prompt = str(messages[0].get('content', ''))[:800]
                response = str(messages[1].get('content', ''))[:800]

                # Quality filters
                if len(prompt) < 10 or len(response) < 10:
                    continue

                # Deduplication
                key = (prompt[:150], response[:150])
                if key in seen:
                    continue
                seen.add(key)

                # Tokenization
                full_text = TextFormat.format_conversation(prompt, response)
                tokens = self.tokenizer.encode(full_text, max_length=self.max_length)

                if len(tokens) >= 20:
                    samples.append(tokens)

            except Exception as e:
                logger.debug(f"Skipping sample: {e}")
                continue

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx], dtype=torch.long)


# ----------------------------
# Batch Collation
# ----------------------------
def collate_fn(batch: List[torch.Tensor], pad_token_id: int = 50256) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collate batch with padding and attention masks"""
    max_len = max(x.size(0) for x in batch)
    padded_batch = []
    masks = []

    for x in batch:
        if x.size(0) < max_len:
            padding = torch.full((max_len - x.size(0),), pad_token_id, dtype=torch.long)
            x_padded = torch.cat([x, padding])
            mask = torch.cat([torch.ones(x.size(0)), torch.zeros(padding.size(0))])
        else:
            x_padded = x
            mask = torch.ones(x.size(0))

        padded_batch.append(x_padded)
        masks.append(mask)

    return torch.stack(padded_batch), torch.stack(masks)


# ----------------------------
# Transformer Model
# ----------------------------
class ProductionTransformer(nn.Module):
    """Production-grade transformer with validation"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        config.validate()  # Ensure valid configuration
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_emb = nn.Embedding(config.max_sequence_length, config.embed_dim)
        self.emb_dropout = nn.Dropout(config.dropout)
        self.emb_norm = nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=config.num_layers)
        self.output_norm = nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps)
        self.output = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        self.output.weight = self.token_emb.weight

        if torch.cuda.is_available():
            for layer in self.transformer.layers:
                layer.gradient_checkpointing = True

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights with proper scaling"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate causal attention mask to prevent looking ahead"""
        mask = torch.zeros(seq_len, seq_len, device=device)
        mask = mask.masked_fill(
            torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool(),
            float('-inf')
        )
        return mask

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through transformer"""
        batch_size, seq_len = x.size()
        device = x.device

        pos = torch.arange(0, seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        token_embeddings = self.token_emb(x)
        pos_embeddings = self.pos_emb(pos)
        x = self.emb_norm(token_embeddings + pos_embeddings)
        x = self.emb_dropout(x)

        causal_mask = self._generate_causal_mask(seq_len, device)
        x = self.transformer(tgt=x, memory=x, tgt_mask=causal_mask, memory_mask=None)
        x = self.output_norm(x)
        logits = self.output(x)

        return logits


# ----------------------------
# Model Validator
# ----------------------------
class ModelValidator:
    """Evaluate model on the validation set"""

    def __init__(self, model: nn.Module, device: torch.device, vocab_size: int, pad_token_id: int):
        self.model = model
        self.device = device
        self.vocab_size = vocab_size
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id, label_smoothing=0.1)

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on the validation set"""
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0

        for batch, masks in tqdm(val_loader, desc="Validating"):
            batch = batch.to(self.device)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]

            logits = self.model(inputs)
            loss = self.criterion(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
            total_loss += loss.item() * targets.numel()
            total_tokens += targets.numel()

        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
        perplexity = np.exp(min(avg_loss, 100))  # Cap to prevent inf

        return {
            'val_loss': avg_loss,
            'val_perplexity': perplexity
        }


# ----------------------------
# Inference Engine
# ----------------------------
class InferenceEngine:
    """High-performance text generation"""

    def __init__(self, model_path: str, device: Optional[str] = None):
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = torch.device(device)
        logger.info(f"Initializing InferenceEngine on device: {self.device}")

        ckpt_path = Path(model_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location=self.device)
        model_config = ModelConfig(**checkpoint['model_config'])
        self.model_config = model_config

        self.tokenizer = ProductionTokenizer()
        self.model = ProductionTransformer(model_config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        logger.info(f"Loaded model from {ckpt_path} on {self.device}")
        params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model parameters: {params:,}")

    @torch.no_grad()
    def generate(
            self,
            prompt: str,
            max_new_tokens: int = 100,
            temperature: float = 0.7,
            top_k: int = 50,
            top_p: float = 0.9,
            repetition_penalty: float = 1.2
    ) -> str:
        """Generate text continuation"""
        self.model.eval()

        formatted_prompt = TextFormat.format_prompt(prompt)
        input_tokens = self.tokenizer.encode(formatted_prompt)
        if not input_tokens:
            return "Unable to process prompt."

        max_ctx = self.model_config.max_sequence_length
        if len(input_tokens) > max_ctx:
            input_tokens = input_tokens[-max_ctx:]

        input_ids = torch.tensor([input_tokens], dtype=torch.long, device=self.device)
        generated_tokens = []
        token_freq = {}

        for _ in range(max_new_tokens):
            logits = self.model(input_ids)[:, -1, :]

            # Apply repetition penalty
            if repetition_penalty != 1.0 and generated_tokens:
                self._apply_repetition_penalty(
                    logits, generated_tokens, token_freq, repetition_penalty
                )

            # Temperature scaling
            if temperature > 0:
                logits = logits / max(temperature, 1e-8)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                self._process_token(next_token, input_ids, generated_tokens, token_freq)
                if self._should_stop(generated_tokens):
                    break
                continue

            # Top-k filtering
            if top_k is not None and top_k > 0:
                logits = self._apply_top_k(logits, top_k)

            # Top-p (nucleus) sampling
            if top_p is not None and 0.0 < top_p < 1.0:
                probs = self._apply_top_p(logits, top_p)
            else:
                probs = F.softmax(logits, dim=-1)

            # Sample next token
            next_token = torch.multinomial(probs, num_samples=1)
            self._process_token(next_token, input_ids, generated_tokens, token_freq)

            if self._should_stop(generated_tokens):
                break

            if input_ids.size(1) > max_ctx:
                input_ids = input_ids[:, -max_ctx:]

        return self._decode_output(input_ids, input_tokens, generated_tokens)

    @staticmethod
    def _apply_repetition_penalty(logits, generated_tokens, token_freq, penalty):
        """Apply penalty to recently generated tokens"""
        for token in set(generated_tokens[-GenerationConfig.REPETITION_PENALTY_WINDOW:]):
            freq = token_freq.get(token, 1)
            penalty_factor = penalty ** (freq / GenerationConfig.REPETITION_PENALTY_DECAY)

            if logits[0, token] < 0:
                logits[0, token] *= penalty_factor
            else:
                logits[0, token] /= penalty_factor

    @staticmethod
    def _apply_top_k(logits, top_k):
        """Filter to top-k tokens"""
        topk_vals, topk_idx = torch.topk(logits, k=min(top_k, logits.size(-1)), dim=-1)
        mask = torch.full_like(logits, float("-inf"))
        mask.scatter_(1, topk_idx, topk_vals)
        return mask

    @staticmethod
    def _apply_top_p(logits, top_p):
        """Apply nucleus (top-p) sampling"""
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        sorted_mask = cumsum > top_p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False
        sorted_probs[sorted_mask] = 0.0

        result = torch.zeros_like(logits).scatter(1, sorted_idx, sorted_probs)
        return result / (result.sum(dim=-1, keepdim=True) + 1e-10)

    @staticmethod
    def _process_token(token_tensor, input_ids, generated_tokens, token_freq):
        """Process and track generated token"""
        token_id = token_tensor.item()
        input_ids = torch.cat([input_ids, token_tensor], dim=1)
        generated_tokens.append(token_id)
        token_freq[token_id] = token_freq.get(token_id, 0) + 1

    @staticmethod
    def _should_stop(generated_tokens):
        """Check early stopping conditions"""
        if not generated_tokens:
            return False

        # Stop on EOS
        if generated_tokens[-1] == 50256:
            return True

        # Stop on repetitive patterns
        if len(generated_tokens) >= GenerationConfig.REPETITION_PATTERN_LENGTH:
            last_n = generated_tokens[-GenerationConfig.REPETITION_PATTERN_LENGTH:]
            if len(set(last_n)) == 1:
                return True

        # Stop if token appears too many times
        if generated_tokens.count(generated_tokens[-1]) > GenerationConfig.MAX_SAME_TOKEN_COUNT:
            return True

        return False

    @staticmethod
    def _decode_output(input_ids, input_tokens, generated_tokens):
        """Decode generated tokens to text"""
        if not generated_tokens:
            return ""

        text = torch.nn.Module().forward.__self__ if False else None
        # Simplified: just decode what we have
        from tiktoken import get_encoding
        enc = get_encoding("gpt2")
        text = enc.decode(generated_tokens)
        text = text.replace(TextFormat.EOS_TOKEN, '').strip()

        if len(set(text)) <= 2 and len(text) > 20:
            return "I'm still learning how to respond properly."

        return text


# ----------------------------
# Training Engine
# ----------------------------
class ProductionTrainer:
    """Production-grade training with validation and monitoring"""

    def __init__(self,
                 model: nn.Module,
                 model_config: ModelConfig,
                 train_config: TrainingConfig,
                 tokenizer: ProductionTokenizer,
                 device: torch.device,
                 output_dir: str = "./checkpoints"):
        self.model = model
        self.model_config = model_config
        self.train_config = train_config
        self.tokenizer = tokenizer
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.validator = ModelValidator(model, device, model_config.vocab_size, tokenizer.pad_token_id)
        self.scaler = torch.cuda.amp.GradScaler() if (train_config.use_fp16 and device.type == "cuda") else None
        self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, label_smoothing=0.1)
        self.global_step = 0

        np.random.seed(train_config.seed)
        torch.manual_seed(train_config.seed)

        try:
            wandb.init(project="production-transformer",
                       config={**model_config.to_dict(), **train_config.to_dict()})
        except Exception as e:
            logger.warning(f"W&B init failed: {e}")

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        """Main training loop"""
        logger.info("Starting training...")
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.train_config.learning_rate,
            weight_decay=self.train_config.weight_decay,
            betas=(0.9, 0.95)
        )

        total_steps = (
                              len(train_loader) // self.train_config.gradient_accumulation_steps) * self.train_config.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.train_config.warmup_steps,
            num_training_steps=total_steps
        )

        self.model.train()

        for epoch in range(self.train_config.num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{self.train_config.num_epochs}")
            self._train_epoch(epoch, train_loader, val_loader, optimizer, scheduler)

        logger.info("Training complete!")
        self.save_checkpoint("final_model.pt")

    def _train_epoch(self, epoch, train_loader, val_loader, optimizer, scheduler):
        """Train for one epoch"""
        epoch_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc="Training")
        for batch_idx, (batch, masks) in enumerate(pbar):
            batch = batch.to(self.device)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]

            ctx = torch.cuda.amp.autocast() if self.scaler else nullcontext()
            with ctx:
                logits = self.model(inputs)
                loss = self.criterion(logits.reshape(-1, self.model_config.vocab_size), targets.reshape(-1))
                loss = loss / self.train_config.gradient_accumulation_steps

            # NaN check
            if batch_idx % self.train_config.nan_check_interval == 0:
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.error(f"NaN/Inf loss detected at step {self.global_step}!")
                    raise RuntimeError("Training diverged - NaN/Inf loss")

            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_config.max_grad_norm)

            if (batch_idx + 1) % self.train_config.gradient_accumulation_steps == 0:
                if self.scaler:
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()
                self.global_step += 1

                # Sample generation
                if self.global_step % self.train_config.sample_generation_steps == 0:
                    self._log_samples()

                # Validation
                if self.global_step % self.train_config.eval_steps == 0 and val_loader:
                    self._validate(val_loader)

                # Checkpointing
                if self.global_step % self.train_config.save_steps == 0:
                    self.save_checkpoint(f"checkpoint_step_{self.global_step}.pt")

            epoch_loss += loss.item() * self.train_config.gradient_accumulation_steps
            pbar.set_postfix({'loss': f'{loss.item() * self.train_config.gradient_accumulation_steps:.4f}'})

        avg_epoch_loss = epoch_loss / len(train_loader)
        logger.info(f"Epoch loss: {avg_epoch_loss:.4f}")

        if wandb.run:
            wandb.log({'epoch_loss': avg_epoch_loss, 'epoch': epoch + 1})

        self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt")

    def _log_samples(self):
        """Generate and log sample outputs"""
        self.model.eval()
        inference_engine = InferenceEngine.__new__(InferenceEngine)
        inference_engine.model = self.model
        inference_engine.model_config = self.model_config
        inference_engine.tokenizer = self.tokenizer
        inference_engine.device = self.device

        test_prompts = ["Hello", "How are you", "Tell me about"]
        logger.info(f"\nStep {self.global_step} - Sample generations:")
        for prompt in test_prompts:
            try:
                sample = inference_engine.generate(prompt, max_new_tokens=50, temperature=0.7)
                logger.info(f"  '{prompt}' -> '{sample}'")
            except Exception as e:
                logger.warning(f"  Failed to generate for '{prompt}': {e}")
        self.model.train()

    def _validate(self, val_loader: DataLoader):
        """Run validation"""
        val_metrics = self.validator.evaluate(val_loader)
        logger.info(f"Step {self.global_step}: {val_metrics}")
        if wandb.run:
            wandb.log({**val_metrics, 'step': self.global_step})
        self.model.train()

    def save_checkpoint(self, filename: str = "model.pt"):
        """Save model checkpoint"""
        path = self.output_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': self.model_config.to_dict(),
            'train_config': self.train_config.to_dict(),
            'global_step': self.global_step
        }, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.global_step = checkpoint.get('global_step', 0)
        logger.info(f"Loaded checkpoint from {path}")


# ----------------------------
# Main Training Script
# ----------------------------
def main():
    """Production training pipeline with automatic checkpoint resume"""

    model_config = ModelConfig(
        embed_dim=768,
        num_heads=12,
        num_layers=12,
        ff_dim=3072,
        max_sequence_length=1024,
        dropout=0.1
    )

    train_config = TrainingConfig(
        batch_size=32,
        gradient_accumulation_steps=2,
        learning_rate=5e-5,
        num_epochs=20,
        warmup_steps=500,
        eval_steps=250,
        save_steps=1000,
        use_fp16=True,
        max_samples_per_dataset=100000,
        sample_generation_steps=100
    )

    # Device selection with auto-tuning
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA - enabling FP16, gradient checkpointing, and GradScaler")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        train_config.use_fp16 = False
        train_config.batch_size = 4
        train_config.gradient_accumulation_steps = 8
        model_config.embed_dim = 512
        model_config.num_heads = 8
        model_config.num_layers = 8
        model_config.ff_dim = 2048
        model_config.max_sequence_length = 512
        logger.info("Using MPS - reduced model (512 embed, 8 heads, 8 layers, seq 512), batch=4, accum=8")
    else:
        device = torch.device("cpu")
        train_config.use_fp16 = False
        train_config.batch_size = 8
        logger.info("Using CPU - disabling FP16")

    logger.info(f"Using device: {device}")

    tokenizer = ProductionTokenizer()
    model = ProductionTransformer(model_config).to(device)

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    cache_dir = f"./dataset_cache_{model_config.max_sequence_length}"

    train_dataset = ProductionConversationDataset(
        tokenizer=tokenizer,
        max_length=model_config.max_sequence_length,
        max_samples=train_config.max_samples_per_dataset,
        cache_dir=cache_dir
    )

    val_dataset = ProductionConversationDataset(
        dataset_name="HuggingFaceH4/ultrachat_200k",
        split="test_sft",
        tokenizer=tokenizer,
        max_length=model_config.max_sequence_length,
        max_samples=5000,
        cache_dir=cache_dir
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_with_pad(batch, pad_token_id=tokenizer.pad_token_id),
        num_workers=0,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_with_pad(batch, pad_token_id=tokenizer.pad_token_id),
        num_workers=0,
        pin_memory=False
    )

    trainer = ProductionTrainer(
        model=model,
        model_config=model_config,
        train_config=train_config,
        tokenizer=tokenizer,
        device=device,
        output_dir="./checkpoints"
    )

    # --- Resume from latest checkpoint if available ---
    checkpoint_dir = Path("./checkpoints")
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_*.pt"), key=os.path.getmtime)
    if checkpoints:
        latest_ckpt = checkpoints[-1]
        logger.info(f"Resuming training from checkpoint: {latest_ckpt}")
        trainer.load_checkpoint(latest_ckpt)

    trainer.train(train_loader, val_loader)

if __name__ == '__main__':
    main()
