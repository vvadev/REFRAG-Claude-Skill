---
name: refrag
description: "Use this skill when implementing REFRAG (REpresentation For RAG) - an efficient decoding framework for retrieval-augmented generation. Triggers: requests to optimize RAG performance, reduce TTFT (time-to-first-token), extend context windows, implement chunk compression for RAG, build selective expansion mechanisms, or create efficient long-context LLM applications. Use when users mention 'REFRAG', 'RAG optimization', 'context compression', 'chunk embeddings', or want to accelerate RAG inference while maintaining accuracy."
---

# REFRAG: Efficient RAG Decoding Framework

## Overview

REFRAG (REpresentation For RAG) is a decoding framework that achieves **30.85× TTFT acceleration** and **16× context extension** for RAG applications without loss in accuracy.

**Core Innovation**: Instead of feeding thousands of raw tokens from retrieved passages, REFRAG compresses chunks into dense embeddings, selectively expands important chunks, and processes the rest in compressed form.

**Key Insight**: In RAG, most retrieved passages exhibit low semantic similarity (block-diagonal attention patterns), making most computations during decoding unnecessary.

## Quick Reference

| Task | Approach | Key Component |
|------|----------|---------------|
| Chunk compression | Split passages into k-token chunks → encode to embeddings | Lightweight encoder (RoBERTa/BERT) |
| Selective expansion | RL policy identifies important chunks | Policy network with REINFORCE |
| Decoding | Process compressed chunks + expanded tokens | Modified decoder with chunk attention |
| Training | Two-phase continual pretraining | Reconstruction → Next-paragraph prediction |
| Retrieval | FAISS index with embeddings | Standard dense retrieval |

---

## Architecture Components

### 1. The Three-Stage Pipeline: Compress → Sense → Expand

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  COMPRESS   │ --> │    SENSE    │ --> │   EXPAND    │
│             │     │             │     │             │
│ Chunk → Emb │     │ RL Policy   │     │ Decoder +   │
│             │     │ Selection   │     │ Generation  │
└─────────────┘     └─────────────┘     └─────────────┘
```

**Compress**: Retrieved passages → Fixed-size chunks (e.g., 16 tokens) → Dense chunk embeddings  
**Sense**: Policy network identifies information-dense chunks for expansion  
**Expand**: Selected chunks feed raw tokens to decoder; others stay compressed

---

## Implementation Guide

### Stage 1: Chunk Compression

**Goal**: Reduce sequence length from N tokens to N/k chunk embeddings (16× reduction with k=16).

#### Encoder Architecture

```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class ChunkEncoder(nn.Module):
    """Compresses token chunks into dense embeddings"""
    
    def __init__(self, encoder_name="roberta-base", hidden_size=768, projection_dim=None):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        
        # Optional projection to decoder space
        self.projection = None
        if projection_dim:
            self.projection = nn.Linear(hidden_size, projection_dim)
    
    def forward(self, chunk_tokens, attention_mask=None):
        """
        Args:
            chunk_tokens: [batch, num_chunks, chunk_size] - tokenized chunks
            attention_mask: [batch, num_chunks, chunk_size]
        Returns:
            chunk_embeddings: [batch, num_chunks, hidden_size]
        """
        batch_size, num_chunks, chunk_size = chunk_tokens.shape
        
        # Flatten for encoding
        flat_tokens = chunk_tokens.view(batch_size * num_chunks, chunk_size)
        flat_mask = attention_mask.view(batch_size * num_chunks, chunk_size) if attention_mask else None
        
        # Encode chunks
        outputs = self.encoder(flat_tokens, attention_mask=flat_mask)
        
        # Use CLS token as chunk representation
        chunk_embeddings = outputs.last_hidden_state[:, 0, :]  # [batch*num_chunks, hidden]
        chunk_embeddings = chunk_embeddings.view(batch_size, num_chunks, -1)
        
        # Optional projection to decoder token space
        if self.projection:
            chunk_embeddings = self.projection(chunk_embeddings)
        
        return chunk_embeddings
```

#### Chunking Strategy

```python
def create_chunks(passages: list[str], chunk_size: int = 16, tokenizer=None):
    """
    Split retrieved passages into fixed-size chunks
    
    Args:
        passages: List of retrieved text passages
        chunk_size: Number of tokens per chunk (k parameter)
        tokenizer: HuggingFace tokenizer
    
    Returns:
        chunks: List of tokenized chunks [num_chunks, chunk_size]
        chunk_metadata: Metadata for reconstruction
    """
    all_chunks = []
    chunk_metadata = []
    
    for passage_idx, passage in enumerate(passages):
        # Tokenize passage
        tokens = tokenizer.encode(passage, add_special_tokens=False)
        
        # Split into chunks
        for i in range(0, len(tokens), chunk_size):
            chunk = tokens[i:i + chunk_size]
            
            # Pad if necessary
            if len(chunk) < chunk_size:
                chunk = chunk + [tokenizer.pad_token_id] * (chunk_size - len(chunk))
            
            all_chunks.append(chunk)
            chunk_metadata.append({
                'passage_idx': passage_idx,
                'chunk_idx': i // chunk_size,
                'start_pos': i,
                'original_length': min(chunk_size, len(tokens) - i)
            })
    
    return torch.tensor(all_chunks), chunk_metadata
```

---

### Stage 2: Selective Expansion (Sense)

**Goal**: Identify which chunks should bypass compression and feed raw tokens to the decoder.

#### Policy Network with REINFORCE

```python
class ExpansionPolicy(nn.Module):
    """RL policy for selective chunk expansion"""
    
    def __init__(self, input_dim=768, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Probability of expansion
        )
    
    def forward(self, chunk_embeddings):
        """
        Args:
            chunk_embeddings: [batch, num_chunks, hidden_dim]
        Returns:
            expansion_probs: [batch, num_chunks] - probability of expanding each chunk
        """
        return self.network(chunk_embeddings).squeeze(-1)
    
    def sample_actions(self, chunk_embeddings, expand_fraction=0.25):
        """
        Sample which chunks to expand using policy probabilities
        
        Args:
            chunk_embeddings: Chunk representations
            expand_fraction: Target fraction of chunks to expand (p parameter)
        Returns:
            actions: [batch, num_chunks] - binary mask (1=expand, 0=compress)
            log_probs: Log probabilities for REINFORCE
        """
        probs = self.forward(chunk_embeddings)
        
        # Sample binary actions
        actions = torch.bernoulli(probs)
        
        # Ensure we don't exceed expand_fraction
        batch_size, num_chunks = probs.shape
        max_expansions = int(num_chunks * expand_fraction)
        
        for b in range(batch_size):
            if actions[b].sum() > max_expansions:
                # Keep only top-k by probability
                _, top_indices = torch.topk(probs[b], max_expansions)
                mask = torch.zeros_like(actions[b])
                mask[top_indices] = 1
                actions[b] = mask
        
        # Compute log probabilities for REINFORCE
        log_probs = actions * torch.log(probs + 1e-10) + (1 - actions) * torch.log(1 - probs + 1e-10)
        
        return actions, log_probs.sum(dim=1)
```

#### PPL-Based Heuristic (Alternative to RL)

```python
def heuristic_expansion(chunk_embeddings, decoder, expand_fraction=0.25):
    """
    Fallback: Use perplexity heuristic instead of RL policy
    Expand chunks with highest predicted perplexity (most uncertain)
    
    Args:
        chunk_embeddings: [batch, num_chunks, hidden_dim]
        decoder: LLM decoder for perplexity estimation
        expand_fraction: Fraction of chunks to expand
    Returns:
        expansion_mask: [batch, num_chunks] - binary selection
    """
    # Compute perplexity scores (simplified)
    with torch.no_grad():
        scores = decoder.estimate_chunk_importance(chunk_embeddings)
    
    # Select top-p chunks by score
    batch_size, num_chunks = chunk_embeddings.shape[:2]
    num_expand = int(num_chunks * expand_fraction)
    
    _, top_indices = torch.topk(scores, num_expand, dim=1)
    expansion_mask = torch.zeros(batch_size, num_chunks, device=chunk_embeddings.device)
    expansion_mask.scatter_(1, top_indices, 1.0)
    
    return expansion_mask
```

---

### Stage 3: Decoder with Mixed Inputs

**Goal**: Process sequence containing both chunk embeddings (compressed) and raw tokens (expanded).

#### Modified Attention Mechanism

```python
class REFRAGDecoder(nn.Module):
    """Decoder that handles mixed chunk embeddings + raw tokens"""
    
    def __init__(self, base_decoder, chunk_token_id=-100):
        super().__init__()
        self.decoder = base_decoder
        self.chunk_token_id = chunk_token_id  # Special token for compressed chunks
    
    def prepare_inputs(self, chunk_embeddings, original_tokens, expansion_mask, chunk_metadata):
        """
        Construct mixed input sequence
        
        Args:
            chunk_embeddings: [batch, num_chunks, hidden_dim]
            original_tokens: Original tokenized passages
            expansion_mask: [batch, num_chunks] - which chunks to expand
            chunk_metadata: Info for reconstruction
        
        Returns:
            mixed_embeddings: Sequence with chunks + expanded tokens
            attention_mask: Corresponding attention mask
        """
        batch_size, num_chunks = expansion_mask.shape
        mixed_sequence = []
        
        for b in range(batch_size):
            seq_parts = []
            for c in range(num_chunks):
                if expansion_mask[b, c] == 1:
                    # Expand: Use raw tokens
                    chunk_start = chunk_metadata[c]['start_pos']
                    chunk_end = chunk_start + chunk_metadata[c]['original_length']
                    tokens = original_tokens[b][chunk_start:chunk_end]
                    token_embeds = self.decoder.get_input_embeddings()(tokens)
                    seq_parts.append(token_embeds)
                else:
                    # Compress: Use chunk embedding
                    seq_parts.append(chunk_embeddings[b, c:c+1])
            
            mixed_sequence.append(torch.cat(seq_parts, dim=0))
        
        # Pad to same length
        max_len = max(seq.shape[0] for seq in mixed_sequence)
        padded = torch.zeros(batch_size, max_len, chunk_embeddings.shape[-1], 
                           device=chunk_embeddings.device)
        attention_mask = torch.zeros(batch_size, max_len, device=chunk_embeddings.device)
        
        for b, seq in enumerate(mixed_sequence):
            seq_len = seq.shape[0]
            padded[b, :seq_len] = seq
            attention_mask[b, :seq_len] = 1
        
        return padded, attention_mask
    
    def forward(self, chunk_embeddings, original_tokens, expansion_mask, chunk_metadata, 
                query_tokens=None):
        """
        Generate response given compressed context + query
        
        Args:
            chunk_embeddings: Compressed chunk representations
            original_tokens: Original passage tokens (for expansion)
            expansion_mask: Which chunks to expand
            chunk_metadata: Chunk reconstruction info
            query_tokens: User query tokens
        
        Returns:
            generated_tokens: Model output
        """
        # Prepare mixed input
        context_embeds, context_mask = self.prepare_inputs(
            chunk_embeddings, original_tokens, expansion_mask, chunk_metadata
        )
        
        # Append query if provided
        if query_tokens is not None:
            query_embeds = self.decoder.get_input_embeddings()(query_tokens)
            full_embeds = torch.cat([context_embeds, query_embeds], dim=1)
            query_mask = torch.ones(query_tokens.shape, device=query_tokens.device)
            full_mask = torch.cat([context_mask, query_mask], dim=1)
        else:
            full_embeds = context_embeds
            full_mask = context_mask
        
        # Generate with decoder
        outputs = self.decoder(
            inputs_embeds=full_embeds,
            attention_mask=full_mask,
            return_dict=True
        )
        
        return outputs
```

---

## Training: Continual Pretraining (CPT)

REFRAG requires two-phase continual pretraining to teach the model to work with chunk embeddings.

### Phase A: Reconstruction Task

**Objective**: Teach decoder to reconstruct original text from chunk embeddings.

```python
def train_reconstruction(encoder, decoder, policy, dataloader, num_steps=300, lr=2e-5):
    """
    Phase A: Reconstruction pretraining
    Freeze policy, train encoder + decoder to reconstruct passages
    
    Loss: Cross-entropy between reconstructed and original text
    """
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()), 
        lr=lr
    )
    
    policy.eval()  # Freeze policy
    encoder.train()
    decoder.train()
    
    for step, batch in enumerate(dataloader):
        if step >= num_steps:
            break
        
        passages = batch['passages']
        
        # Create chunks and encode
        chunk_tokens, metadata = create_chunks(passages, chunk_size=16)
        chunk_embeddings = encoder(chunk_tokens)
        
        # Reconstruction: Use all chunks compressed (no expansion yet)
        expansion_mask = torch.zeros(chunk_embeddings.shape[:2])
        
        # Forward pass
        outputs = decoder(
            chunk_embeddings=chunk_embeddings,
            original_tokens=chunk_tokens,
            expansion_mask=expansion_mask,
            chunk_metadata=metadata,
            query_tokens=None
        )
        
        # Reconstruction loss
        target_tokens = torch.cat([chunk for chunk in chunk_tokens], dim=1)
        loss = torch.nn.functional.cross_entropy(
            outputs.logits.view(-1, outputs.logits.shape[-1]),
            target_tokens.view(-1)
        )
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 20 == 0:
            print(f"Step {step}, Reconstruction Loss: {loss.item():.4f}")
```

### Phase B: Next-Paragraph Prediction with Selective Expansion

**Objective**: Teach policy to select important chunks while maintaining generation quality.

```python
def train_next_paragraph(encoder, decoder, policy, dataloader, 
                        num_steps=300, lr=2e-5, expand_fraction=0.25):
    """
    Phase B: Next-paragraph prediction with selective expansion
    Unfreeze all components, train end-to-end with RL
    
    Reward: Negative perplexity of next paragraph prediction
    """
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()) + list(policy.parameters()),
        lr=lr
    )
    
    encoder.train()
    decoder.train()
    policy.train()
    
    for step, batch in enumerate(dataloader):
        if step >= num_steps:
            break
        
        context_passages = batch['context']
        target_paragraphs = batch['target']
        
        # Encode context chunks
        chunk_tokens, metadata = create_chunks(context_passages, chunk_size=16)
        chunk_embeddings = encoder(chunk_tokens)
        
        # Sample expansion policy
        expansion_mask, log_probs = policy.sample_actions(
            chunk_embeddings, 
            expand_fraction=expand_fraction
        )
        
        # Generate prediction
        outputs = decoder.generate(
            chunk_embeddings=chunk_embeddings,
            original_tokens=chunk_tokens,
            expansion_mask=expansion_mask,
            chunk_metadata=metadata,
            max_new_tokens=128
        )
        
        # Compute reward (negative perplexity on target)
        with torch.no_grad():
            target_loss = decoder.compute_loss(outputs, target_paragraphs)
            reward = -target_loss  # Higher reward for lower perplexity
        
        # REINFORCE policy gradient
        policy_loss = -(log_probs * reward).mean()
        
        # Reconstruction regularization
        recon_loss = decoder.compute_reconstruction_loss(
            chunk_embeddings, expansion_mask, chunk_tokens
        )
        
        total_loss = policy_loss + 0.1 * recon_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if step % 20 == 0:
            print(f"Step {step}, Policy Loss: {policy_loss.item():.4f}, "
                  f"Reward: {reward.mean().item():.4f}")
```

---

## End-to-End RAG Pipeline

### Complete REFRAG System

```python
import faiss
import numpy as np

class REFRAGSystem:
    """Complete REFRAG RAG system"""
    
    def __init__(self, encoder, decoder, policy, retriever_model, 
                 chunk_size=16, expand_fraction=0.25):
        self.encoder = encoder
        self.decoder = decoder
        self.policy = policy
        self.retriever = retriever_model
        self.chunk_size = chunk_size
        self.expand_fraction = expand_fraction
        self.index = None
        self.passages = []
    
    def build_index(self, corpus: list[str], index_path: str = None):
        """
        Build FAISS index for retrieval
        
        Args:
            corpus: List of passages to index
            index_path: Optional path to save index
        """
        # Encode passages for retrieval
        embeddings = []
        for passage in corpus:
            emb = self.retriever.encode(passage)
            embeddings.append(emb)
        
        embeddings = np.array(embeddings).astype('float32')
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for similarity
        self.index.add(embeddings)
        self.passages = corpus
        
        if index_path:
            faiss.write_index(self.index, index_path)
        
        print(f"Indexed {len(corpus)} passages")
    
    def retrieve(self, query: str, top_k: int = 5):
        """
        Retrieve top-k relevant passages
        
        Args:
            query: Search query
            top_k: Number of passages to retrieve
        Returns:
            retrieved_passages: List of top-k passages
        """
        # Encode query
        query_emb = self.retriever.encode(query).reshape(1, -1).astype('float32')
        
        # Search
        scores, indices = self.index.search(query_emb, top_k)
        retrieved = [self.passages[idx] for idx in indices[0]]
        
        return retrieved
    
    def generate(self, query: str, top_k: int = 5, max_new_tokens: int = 128):
        """
        Full RAG generation with REFRAG
        
        Args:
            query: User query
            top_k: Number of passages to retrieve
            max_new_tokens: Maximum tokens to generate
        Returns:
            response: Generated text
            metadata: Timing and compression stats
        """
        import time
        
        # 1. Retrieval
        start = time.time()
        passages = self.retrieve(query, top_k)
        retrieval_time = time.time() - start
        
        # 2. Compress: Chunk and encode passages
        start = time.time()
        chunk_tokens, metadata = create_chunks(
            passages, 
            chunk_size=self.chunk_size,
            tokenizer=self.encoder.tokenizer
        )
        chunk_embeddings = self.encoder(chunk_tokens)
        compression_time = time.time() - start
        
        # 3. Sense: Select chunks to expand
        start = time.time()
        with torch.no_grad():
            expansion_mask, _ = self.policy.sample_actions(
                chunk_embeddings,
                expand_fraction=self.expand_fraction
            )
        selection_time = time.time() - start
        
        # 4. Expand & Generate
        start = time.time()
        query_tokens = self.encoder.tokenizer.encode(query, return_tensors='pt')
        
        outputs = self.decoder.generate(
            chunk_embeddings=chunk_embeddings,
            original_tokens=chunk_tokens,
            expansion_mask=expansion_mask,
            chunk_metadata=metadata,
            query_tokens=query_tokens,
            max_new_tokens=max_new_tokens
        )
        
        response = self.encoder.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generation_time = time.time() - start
        
        # Statistics
        num_chunks = chunk_embeddings.shape[1]
        num_expanded = expansion_mask.sum().item()
        compression_ratio = self.chunk_size  # Original tokens per chunk
        effective_length = num_expanded * self.chunk_size + (num_chunks - num_expanded)
        
        stats = {
            'retrieval_time': retrieval_time,
            'compression_time': compression_time,
            'selection_time': selection_time,
            'generation_time': generation_time,
            'total_time': retrieval_time + compression_time + selection_time + generation_time,
            'num_chunks': num_chunks,
            'num_expanded': num_expanded,
            'expansion_rate': num_expanded / num_chunks,
            'effective_sequence_length': effective_length,
            'compression_ratio': (num_chunks * self.chunk_size) / effective_length
        }
        
        return response, stats
```

---

## Usage Examples

### Example 1: Basic Setup

```python
from transformers import AutoModel, AutoTokenizer, LlamaForCausalLM

# Initialize components
encoder = ChunkEncoder(encoder_name="roberta-base", projection_dim=4096)
base_decoder = LlamaForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B")
decoder = REFRAGDecoder(base_decoder)
policy = ExpansionPolicy(input_dim=768)

# Retriever for passage encoding
from sentence_transformers import SentenceTransformer
retriever = SentenceTransformer('BAAI/bge-small-en-v1.5')

# Create REFRAG system
system = REFRAGSystem(
    encoder=encoder,
    decoder=decoder,
    policy=policy,
    retriever_model=retriever,
    chunk_size=16,
    expand_fraction=0.25
)

# Build index
corpus = [
    "Paris is the capital of France...",
    "The Eiffel Tower was built in 1889...",
    # ... more passages
]
system.build_index(corpus, index_path="data/index.faiss")
```

### Example 2: Generate with REFRAG

```python
# Query the system
query = "When was the Eiffel Tower built?"
response, stats = system.generate(
    query=query,
    top_k=4,
    max_new_tokens=128
)

print(f"Response: {response}")
print(f"\nStatistics:")
print(f"  Total time: {stats['total_time']:.3f}s")
print(f"  Compression ratio: {stats['compression_ratio']:.2f}x")
print(f"  Chunks: {stats['num_chunks']} (expanded: {stats['num_expanded']})")
```

### Example 3: Training Pipeline

```python
from torch.utils.data import DataLoader

# Prepare training data
train_dataset = [
    {'passages': [...], 'context': [...], 'target': "..."},
    # ... more examples
]
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# Phase A: Reconstruction
print("Phase A: Reconstruction training...")
train_reconstruction(
    encoder=encoder,
    decoder=decoder,
    policy=policy,
    dataloader=train_loader,
    num_steps=300,
    lr=2e-5
)

# Phase B: Next-paragraph with RL
print("Phase B: Next-paragraph prediction...")
train_next_paragraph(
    encoder=encoder,
    decoder=decoder,
    policy=policy,
    dataloader=train_loader,
    num_steps=300,
    lr=2e-5,
    expand_fraction=0.25
)

# Save trained model
torch.save({
    'encoder': encoder.state_dict(),
    'decoder': decoder.state_dict(),
    'policy': policy.state_dict()
}, 'refrag_model.pt')
```

---

## Key Hyperparameters

| Parameter | Symbol | Typical Value | Description |
|-----------|--------|---------------|-------------|
| Chunk size | k | 16, 32 | Tokens per chunk (higher k = more compression) |
| Expansion fraction | p | 0.25 | Fraction of chunks to expand (0-1) |
| Context max | ctx_max | 1024, 2048 | Maximum context tokens |
| Learning rate | lr | 2e-5 | For continual pretraining |
| Training steps | - | 300 | Per phase (reconstruction, next-para) |

**Performance vs. Compression Trade-off**:
- k=16: ~16× compression, 16.53× TTFT speedup
- k=32: ~32× compression, 30.85× TTFT speedup
- p=0.25: Expand 25% of chunks (balance quality/speed)

---

## Performance Metrics

### Measuring REFRAG Effectiveness

```python
import time

def benchmark_refrag(system, queries, ground_truth=None):
    """
    Benchmark REFRAG system performance
    
    Returns:
        metrics: Dict with TTFT, throughput, accuracy
    """
    results = {
        'ttft': [],          # Time to first token
        'ttit': [],          # Time to incremental tokens
        'throughput': [],    # Tokens per second
        'compression_ratio': [],
        'accuracy': []       # If ground_truth provided
    }
    
    for query in queries:
        response, stats = system.generate(query, max_new_tokens=128)
        
        # TTFT = retrieval + compression + selection
        ttft = stats['retrieval_time'] + stats['compression_time'] + stats['selection_time']
        results['ttft'].append(ttft)
        
        # Throughput
        num_tokens = len(system.encoder.tokenizer.encode(response))
        throughput = num_tokens / stats['generation_time']
        results['throughput'].append(throughput)
        
        results['compression_ratio'].append(stats['compression_ratio'])
        
        # Optional: Accuracy evaluation
        if ground_truth:
            # Implement your accuracy metric (e.g., ROUGE, exact match)
            pass
    
    # Aggregate
    return {
        'avg_ttft': np.mean(results['ttft']),
        'avg_throughput': np.mean(results['throughput']),
        'avg_compression': np.mean(results['compression_ratio']),
        'ttft_std': np.std(results['ttft'])
    }
```

---

## Common Patterns

### Pattern 1: Plug-and-Play Integration

REFRAG doesn't modify the base LLM architecture - it operates as a preprocessing layer.

```python
# Use with any decoder-only LLM
def integrate_with_any_llm(llm_name: str):
    """
    REFRAG works with LLaMA, GPT, OPT, Falcon, etc.
    Just wrap the base model in REFRAGDecoder
    """
    base_model = AutoModelForCausalLM.from_pretrained(llm_name)
    refrag_decoder = REFRAGDecoder(base_model)
    return refrag_decoder

# Examples
llama_refrag = integrate_with_any_llm("meta-llama/Llama-3.2-3B")
opt_refrag = integrate_with_any_llm("facebook/opt-6.7b")
```

### Pattern 2: Caching Chunk Embeddings

Pre-compute and cache chunk embeddings for frequently accessed passages.

```python
import pickle

class CachedREFRAG(REFRAGSystem):
    """REFRAG with chunk embedding cache"""
    
    def __init__(self, *args, cache_path="chunk_cache.pkl", **kwargs):
        super().__init__(*args, **kwargs)
        self.cache_path = cache_path
        self.chunk_cache = self.load_cache()
    
    def load_cache(self):
        try:
            with open(self.cache_path, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return {}
    
    def save_cache(self):
        with open(self.cache_path, 'wb') as f:
            pickle.dump(self.chunk_cache, f)
    
    def get_or_compute_chunks(self, passage_id, passage_text):
        """Get cached chunks or compute and cache"""
        if passage_id in self.chunk_cache:
            return self.chunk_cache[passage_id]
        
        # Compute chunks
        chunk_tokens, metadata = create_chunks([passage_text], self.chunk_size)
        chunk_embeddings = self.encoder(chunk_tokens)
        
        # Cache
        self.chunk_cache[passage_id] = {
            'tokens': chunk_tokens,
            'embeddings': chunk_embeddings,
            'metadata': metadata
        }
        
        return self.chunk_cache[passage_id]
```

### Pattern 3: Multi-Turn Conversations

REFRAG excels at maintaining long conversation history.

```python
class ConversationalREFRAG(REFRAGSystem):
    """REFRAG for multi-turn dialogue"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conversation_history = []
    
    def chat(self, user_message: str, retrieve_context: bool = True):
        """
        Multi-turn conversation with context compression
        
        Args:
            user_message: Current user message
            retrieve_context: Whether to retrieve external passages
        """
        # Add to history
        self.conversation_history.append({
            'role': 'user',
            'content': user_message
        })
        
        # Build context from history
        context_passages = []
        if retrieve_context:
            # Retrieve external knowledge
            context_passages.extend(self.retrieve(user_message, top_k=3))
        
        # Add conversation history as passages
        for turn in self.conversation_history[-5:]:  # Last 5 turns
            context_passages.append(f"{turn['role']}: {turn['content']}")
        
        # Generate with REFRAG compression
        response, stats = self.generate_from_passages(
            query=user_message,
            passages=context_passages,
            max_new_tokens=128
        )
        
        # Add to history
        self.conversation_history.append({
            'role': 'assistant',
            'content': response
        })
        
        return response, stats
    
    def generate_from_passages(self, query, passages, max_new_tokens):
        """Generate without retrieval (use provided passages)"""
        # Compress passages
        chunk_tokens, metadata = create_chunks(passages, self.chunk_size)
        chunk_embeddings = self.encoder(chunk_tokens)
        
        # Select important chunks
        expansion_mask, _ = self.policy.sample_actions(chunk_embeddings, self.expand_fraction)
        
        # Generate
        query_tokens = self.encoder.tokenizer.encode(query, return_tensors='pt')
        outputs = self.decoder.generate(
            chunk_embeddings=chunk_embeddings,
            original_tokens=chunk_tokens,
            expansion_mask=expansion_mask,
            chunk_metadata=metadata,
            query_tokens=query_tokens,
            max_new_tokens=max_new_tokens
        )
        
        response = self.encoder.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response, {'num_chunks': len(metadata), 'expanded': expansion_mask.sum().item()}
```

---

## Troubleshooting

### Issue 1: Poor Expansion Policy Performance

**Symptom**: Policy selects irrelevant chunks, degrading accuracy.

**Solutions**:
1. Use PPL heuristic instead of RL during early training
2. Increase `expand_fraction` (e.g., 0.25 → 0.4)
3. Add stronger reward shaping in REINFORCE
4. Pre-train policy on supervised chunk importance labels

```python
# Fallback to heuristic
expansion_mask = heuristic_expansion(chunk_embeddings, decoder, expand_fraction=0.3)
```

### Issue 2: KV Cache Memory Issues

**Symptom**: OOM errors with long contexts even with REFRAG.

**Solutions**:
1. Reduce chunk size k (more compression)
2. Lower expand_fraction
3. Use gradient checkpointing
4. Implement streaming generation

```python
# Enable gradient checkpointing
decoder.decoder.gradient_checkpointing_enable()
```

### Issue 3: Validation Errors

**Symptom**: Sequence length mismatches or attention mask errors.

**Solutions**:
1. Ensure consistent padding across chunks
2. Verify expansion_mask shape matches chunk_embeddings
3. Check metadata indices are valid

```python
# Debug shapes
print(f"Chunks: {chunk_embeddings.shape}")
print(f"Mask: {expansion_mask.shape}")
print(f"Tokens: {chunk_tokens.shape}")
assert chunk_embeddings.shape[1] == expansion_mask.shape[1], "Shape mismatch!"
```

---

## Best Practices

1. **Start with k=16, p=0.25**: Balanced compression and quality
2. **Use curriculum learning**: Reconstruction → Next-paragraph (don't skip Phase A)
3. **Cache chunk embeddings**: Pre-compute for static corpora
4. **Monitor expansion rates**: Track which chunks are selected
5. **Validate on weak retrieval**: REFRAG shines when many passages are irrelevant
6. **Benchmark TTFT**: Measure actual latency improvements
7. **Tune expand_fraction per task**: More expansion for complex reasoning

---

## References

- **Paper**: "REFRAG: Rethinking RAG based Decoding" (arXiv:2509.01092)
- **Authors**: Xiaoqiang Lin, Aritra Ghosh, Bryan Kian Hsiang Low, Anshumali Shrivastava, Vijai Mohan
- **GitHub Implementations**: 
  - https://github.com/simulanics/REFRAG (Full reference)
  - https://github.com/Shaivpidadi/refrag (Simplified)
- **Key Results**: 30.85× TTFT speedup, 16× context extension, no accuracy loss

---

## Advanced Topics

### GPU Optimization

```python
# Mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    outputs = decoder(chunk_embeddings, ...)
    loss = compute_loss(outputs)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Distributed Inference

```python
# Model parallelism for large decoders
from torch.nn.parallel import DistributedDataParallel

encoder = DistributedDataParallel(encoder, device_ids=[local_rank])
decoder = DistributedDataParallel(decoder, device_ids=[local_rank])
```

### Custom Chunk Sizes per Passage

```python
# Variable chunk sizes based on passage importance
def adaptive_chunking(passages, importance_scores, min_k=8, max_k=32):
    """
    Use smaller chunks (more granular) for important passages
    Use larger chunks (more compression) for less important passages
    """
    chunks = []
    for passage, importance in zip(passages, importance_scores):
        k = int(min_k + (max_k - min_k) * (1 - importance))
        passage_chunks = create_chunks([passage], chunk_size=k)
        chunks.append(passage_chunks)
    return chunks
```

---

**Remember**: REFRAG's power comes from exploiting RAG-specific sparsity patterns. It's optimized for scenarios where most retrieved content is irrelevant but you need to process large volumes to find the signal. For tasks where all context is equally important, traditional long-context methods may be more suitable.