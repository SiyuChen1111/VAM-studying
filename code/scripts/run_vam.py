#!/usr/bin/env python3
"""
Quick start script for running original VAM model
"""

import os
import subprocess
import sys

print("="*80)
print("VAM Original Model - Quick Start")
print("="*80)
print()

# Check if we're in the right directory
current_dir = os.getcwd()
if not current_dir.endswith('VAM-studying'):
    print("❌ Please run this script from the VAM-studying directory")
    sys.exit(1)

# Check if vam directory exists
if not os.path.exists('vam'):
    print("❌ vam/ directory not found")
    sys.exit(1)

print("✓ Directory structure verified")
print()

# Check for data files
data_files = [
    'vam/gameplay_data.zip',
    'vam/metadata.csv',
    'vam/graphics.zip',
]

print("Checking data files:")
for f in data_files:
    if os.path.exists(f):
        print(f"  ✓ {f}")
    else:
        print(f"  ⚠ {f} (not found)")
print()

# Check for bird images
bird_images = ['vam/bird0.png', 'vam/bird1.png', 'vam/bird2.png', 'vam/bird3.png']
print("Checking bird images:")
for f in bird_images:
    if os.path.exists(f):
        print(f"  ✓ {f}")
    else:
        print(f"  ⚠ {f} (not found)")
print()

# Check for background
if os.path.exists('vam/bkgrnd.png'):
    print("  ✓ vam/bkgrnd.png")
else:
    print("  ⚠ vam/bkgrnd.png (not found)")
print()

print("="*80)
print("Ready to run original VAM!")
print("="*80)
print()

print("Choose training mode:")
print("  1. Quick test (5 epochs)")
print("  2. Standard training (30 epochs)")
print("  3. Task-optimized model")
print("  4. Binned RT model")
print("  5. Custom configuration")
print()

choice = input("Enter choice (1-5): ").strip()

if choice == '1':
    print("\n🚀 Running quick test (5 epochs)...")
    cmd = [
        sys.executable, '-m', 'vam.training',
        '--project', 'test',
        '--expt_name', 'quick_test',
        '--n_epochs', '5'
    ]
elif choice == '2':
    print("\n🚀 Running standard training (30 epochs)...")
    cmd = [
        sys.executable, '-m', 'vam.training',
        '--project', 'test',
        '--expt_name', 'standard_run',
        '--n_epochs', '30'
    ]
elif choice == '3':
    print("\n🚀 Running task-optimized model...")
    cmd = [
        sys.executable, '-m', 'vam.training',
        '--model_type', 'task_opt',
        '--n_epochs', '30'
    ]
elif choice == '4':
    print("\n🚀 Running binned RT model...")
    cmd = [
        sys.executable, '-m', 'vam.training',
        '--model_type', 'binned_rt',
        '--n_rt_bins', '5',
        '--rt_bin', '3',
        '--n_epochs', '30'
    ]
elif choice == '5':
    print("\n📝 Custom configuration mode")
    project = input("Project name (default: test): ").strip() or 'test'
    expt_name = input("Experiment name (default: custom_run): ").strip() or 'custom_run'
    n_epochs = input("Number of epochs (default: 20): ").strip() or '20'
    model_type = input("Model type (vam/task_opt/binned_rt, default: vam): ").strip() or 'vam'
    
    cmd = [
        sys.executable, '-m', 'vam.training',
        '--project', project,
        '--expt_name', expt_name,
        '--n_epochs', n_epochs,
        '--model_type', model_type
    ]
else:
    print("❌ Invalid choice")
    sys.exit(1)

print("\nCommand to run:")
print(" ".join(cmd))
print()

confirm = input("Run now? (y/n): ").strip().lower()

if confirm == 'y':
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80)
    print()
    
    # Change to vam directory
    os.chdir('vam')
    
    # Run the command
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user")
        sys.exit(0)
    
    print("\n" + "="*80)
    print("✓ Training completed!")
    print("="*80)
else:
    print("\n⚠ Training cancelled")
    print("\nYou can run the command manually:")
    print("cd vam && " + " ".join(cmd))
