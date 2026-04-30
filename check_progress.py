"""
Simple script to check training progress.
Run this in a separate terminal while training is running.
"""

import os
import json
import time

def check_progress():
    output_dir = 'outputs/stage_a_quick'
    models = ['relu_baseline', 'siren_(data_only)', 'siren_+_poisson_(pinn)']
    
    print("\n" + "=" * 70)
    print("TRAINING PROGRESS CHECK")
    print("=" * 70)
    
    for model_name in models:
        model_dir = os.path.join(output_dir, model_name.lower().replace(' ', '_').replace('(', '').replace(')', ''))
        history_file = os.path.join(model_dir, 'training_history.json')
        
        print(f"\n{model_name}:")
        
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                history = json.load(f)
            
            if history['step']:
                last_step = history['step'][-1]
                last_loss = history['loss_total'][-1]
                last_val = history['val_loss_data'][-1]
                print(f"  ✓ Completed - Step {last_step}/5000")
                print(f"    Final loss: {last_loss:.4f}, Val loss: {last_val:.4f}")
            else:
                print(f"  ⏳ Training started but no steps logged yet")
        else:
            # Check for checkpoints
            checkpoints = [f for f in os.listdir(model_dir) if f.startswith('checkpoint_')] if os.path.exists(model_dir) else []
            if checkpoints:
                print(f"  ⏳ Training in progress ({len(checkpoints)} checkpoints saved)")
            elif os.path.exists(model_dir):
                print(f"  ⏳ Training in progress (no checkpoints yet)")
            else:
                print(f"  ⏸️  Not started yet")
    
    print("\n" + "=" * 70)

if __name__ == '__main__':
    check_progress()
