"""
Quick test version of Stage A with reduced training steps for local testing.
This will run much faster (~5-10 minutes on CPU) but with lower accuracy.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import everything from the main script
from run_stage_a import *

# Override configuration for quick testing
CFG.update({
    # Reduced dataset size
    'n_train'     : 50_000,   # Down from 200k
    'n_test'      : 5_000,    # Down from 20k
    
    # Smaller network
    'hidden_features' : 128,  # Down from 256
    'hidden_layers'   : 3,    # Down from 4
    
    # Much shorter training
    'total_steps'     : 5_000,   # Down from 30k
    'warmup_steps'    : 500,     # Down from 3k
    'batch_size'      : 2048,    # Down from 4096
    'coloc_size'      : 2048,    # Down from 4096
    'log_every'       : 500,     # More frequent logging
    'save_every'      : 2_000,   # More frequent saves
    
    'output_dir'      : 'outputs/stage_a_quick',
})

if __name__ == '__main__':
    print("=" * 60)
    print("QUICK TEST VERSION - Reduced training for faster results")
    print("This will take ~5-10 minutes on CPU, ~2-3 minutes on GPU")
    print("=" * 60)
    main()
