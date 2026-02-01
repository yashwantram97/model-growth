"""Main training script for Dense → MoE transition experiment."""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from pathlib import Path
import json
from tqdm import tqdm

from config import ModelConfig, TrainingConfig
from models import DenseTransformer, MoETransformer
from transfer import transfer_dense_to_moe, verify_functional_identity, analyze_expert_diversity
from utils import create_dummy_dataloader, infinite_dataloader


class Trainer:
    """Trainer for Dense → MoE transition experiment."""
    
    def __init__(self, model_config: ModelConfig, training_config: TrainingConfig):
        self.model_config = model_config
        self.training_config = training_config
        
        # Setup device
        self.device = torch.device(training_config.device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Create checkpoint directory
        Path(training_config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        # Training history
        self.history = {
            'dense_losses': [],
            'moe_losses': [],
            'dense_steps': [],
            'moe_steps': []
        }
    
    def train_dense_phase(self):
        """Phase 1: Train dense model for specified steps."""
        print("\n" + "=" * 80)
        print("PHASE 1: Training Dense Model")
        print("=" * 80)
        
        # Create dense model
        dense_model = DenseTransformer(self.model_config).to(self.device)
        print(f"Dense model parameters: {dense_model.get_num_params():,}")
        
        # Create dataloader
        dataloader = create_dummy_dataloader(
            self.model_config.vocab_size,
            self.training_config.seq_length,
            self.training_config.batch_size
        )
        data_iter = infinite_dataloader(dataloader)
        
        # Optimizer
        optimizer = AdamW(
            dense_model.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay
        )
        
        # Training loop
        dense_model.train()
        pbar = tqdm(range(self.training_config.dense_steps), desc="Dense Training")
        
        for step in pbar:
            # Get batch
            batch = next(data_iter).to(self.device)
            
            # Forward pass
            logits = dense_model(batch)
            
            # Compute loss (next token prediction)
            # Shift logits and labels for language modeling
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = batch[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dense_model.parameters(), 1.0)
            optimizer.step()
            
            # Log
            self.history['dense_losses'].append(loss.item())
            self.history['dense_steps'].append(step)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            if (step + 1) % self.training_config.log_interval == 0:
                avg_loss = sum(self.history['dense_losses'][-self.training_config.log_interval:]) / self.training_config.log_interval
                print(f"\nStep {step + 1}/{self.training_config.dense_steps} - Avg Loss: {avg_loss:.4f}")
        
        final_loss = loss.item()
        print(f"\n✓ Dense training completed. Final loss: {final_loss:.4f}")
        
        # Save checkpoint
        checkpoint_path = Path(self.training_config.checkpoint_dir) / "dense_model.pt"
        torch.save({
            'model_state_dict': dense_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': self.model_config,
            'final_loss': final_loss
        }, checkpoint_path)
        print(f"✓ Saved dense model to {checkpoint_path}")
        
        return dense_model, final_loss
    
    def transfer_phase(self, dense_model):
        """Phase 2: Transfer weights from dense to MoE."""
        print("\n" + "=" * 80)
        print("PHASE 2: Weight Transfer (Dense → MoE)")
        print("=" * 80)
        
        # Create MoE model
        moe_model = MoETransformer(self.model_config).to(self.device)
        print(f"MoE model total parameters: {moe_model.get_num_params():,}")
        print(f"MoE active parameters/token: {moe_model.get_active_params_per_token():,}")
        
        # Transfer weights
        moe_model = transfer_dense_to_moe(dense_model, moe_model, verbose=True)
        
        # Verify functional identity
        test_input = torch.randint(
            0, self.model_config.vocab_size,
            (2, self.training_config.seq_length),
            device=self.device
        )
        
        is_preserved, max_diff = verify_functional_identity(
            dense_model, moe_model, test_input, verbose=True
        )
        
        # Analyze initial expert usage
        analyze_expert_diversity(moe_model, test_input, verbose=True)
        
        # Save MoE model
        checkpoint_path = Path(self.training_config.checkpoint_dir) / "moe_model_init.pt"
        torch.save({
            'model_state_dict': moe_model.state_dict(),
            'config': self.model_config,
            'functional_identity_preserved': is_preserved,
            'max_diff': max_diff
        }, checkpoint_path)
        print(f"✓ Saved initial MoE model to {checkpoint_path}")
        
        return moe_model
    
    def train_moe_phase(self, moe_model, transition_loss: float):
        """Phase 3: Train MoE model for specified steps."""
        print("\n" + "=" * 80)
        print("PHASE 3: Training MoE Model")
        print("=" * 80)
        
        # Create dataloader
        dataloader = create_dummy_dataloader(
            self.model_config.vocab_size,
            self.training_config.seq_length,
            self.training_config.batch_size
        )
        data_iter = infinite_dataloader(dataloader)
        
        # Optimizer
        optimizer = AdamW(
            moe_model.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay
        )
        
        # Training loop
        moe_model.train()
        pbar = tqdm(range(self.training_config.moe_steps), desc="MoE Training")
        
        loss_spike_detected = False
        
        for step in pbar:
            # Get batch
            batch = next(data_iter).to(self.device)
            
            # Forward pass
            logits, aux_loss = moe_model(batch)
            
            # Compute loss
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = batch[:, 1:].contiguous()
            ce_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
            # Total loss (CE + load balancing auxiliary loss)
            loss = ce_loss + aux_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(moe_model.parameters(), 1.0)
            optimizer.step()
            
            # Log
            self.history['moe_losses'].append(ce_loss.item())
            self.history['moe_steps'].append(step)
            
            pbar.set_postfix({
                'loss': f'{ce_loss.item():.4f}',
                'aux': f'{aux_loss.item():.4f}'
            })
            
            # Check for loss spike
            if step == 0 and ce_loss.item() > transition_loss * 1.5:
                loss_spike_detected = True
                print(f"\n⚠ WARNING: Loss spike detected! Initial MoE loss {ce_loss.item():.4f} > Dense loss {transition_loss:.4f}")
            
            if (step + 1) % self.training_config.log_interval == 0:
                avg_loss = sum(self.history['moe_losses'][-self.training_config.log_interval:]) / self.training_config.log_interval
                print(f"\nStep {step + 1}/{self.training_config.moe_steps} - Avg Loss: {avg_loss:.4f} (CE: {ce_loss.item():.4f}, Aux: {aux_loss.item():.4f})")
        
        final_loss = ce_loss.item()
        print(f"\n✓ MoE training completed. Final loss: {final_loss:.4f}")
        
        if not loss_spike_detected:
            print("✓ No loss spike detected during transition!")
        
        # Save final MoE model
        checkpoint_path = Path(self.training_config.checkpoint_dir) / "moe_model_final.pt"
        torch.save({
            'model_state_dict': moe_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': self.model_config,
            'final_loss': final_loss,
            'loss_spike_detected': loss_spike_detected
        }, checkpoint_path)
        print(f"✓ Saved final MoE model to {checkpoint_path}")
        
        return moe_model, final_loss, loss_spike_detected
    
    def run_full_experiment(self):
        """Run complete experiment: Dense → Transfer → MoE."""
        print("\n" + "=" * 80)
        print("DENSE → MoE TRANSITION EXPERIMENT")
        print("=" * 80)
        print(f"Model: {self.model_config.d_model}d × {self.model_config.n_layers}L × {self.model_config.n_heads}H")
        print(f"MoE: {self.model_config.num_experts} experts, top-{self.model_config.top_k} routing")
        print(f"Training: {self.training_config.dense_steps} dense steps + {self.training_config.moe_steps} MoE steps")
        
        # Phase 1: Train dense
        dense_model, dense_final_loss = self.train_dense_phase()
        
        # Phase 2: Transfer
        moe_model = self.transfer_phase(dense_model)
        
        # Phase 3: Train MoE
        moe_model, moe_final_loss, loss_spike = self.train_moe_phase(moe_model, dense_final_loss)
        
        # Save training history
        history_path = Path(self.training_config.checkpoint_dir) / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"\n✓ Saved training history to {history_path}")
        
        # Final summary
        print("\n" + "=" * 80)
        print("EXPERIMENT SUMMARY")
        print("=" * 80)
        print(f"Dense final loss:     {dense_final_loss:.4f}")
        print(f"MoE final loss:       {moe_final_loss:.4f}")
        print(f"Loss spike detected:  {'Yes ⚠' if loss_spike else 'No ✓'}")
        print(f"Loss improvement:     {((dense_final_loss - moe_final_loss) / dense_final_loss * 100):.2f}%")
        print("=" * 80 + "\n")
        
        return dense_model, moe_model


def main():
    """Main entry point."""
    # Configuration
    model_config = ModelConfig(
        vocab_size=50257,
        d_model=512,  # Smaller for faster training
        n_layers=6,   # Fewer layers for faster training
        n_heads=8,
        d_ff=2048,
        max_seq_len=512,
        dropout=0.1,
        num_experts=8,
        top_k=2
    )
    
    training_config = TrainingConfig(
        dense_steps=1000,
        moe_steps=1000,
        learning_rate=1e-4,
        weight_decay=0.01,
        batch_size=4,
        seq_length=128,
        log_interval=100,
        device="cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir="./checkpoints"
    )
    
    # Run experiment
    trainer = Trainer(model_config, training_config)
    dense_model, moe_model = trainer.run_full_experiment()
    
    print("✓ Experiment completed successfully!")


if __name__ == "__main__":
    main()
