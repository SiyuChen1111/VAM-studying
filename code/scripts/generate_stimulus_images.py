"""
Generate unique stimulus images for LIM task.

This script generates the 28 unique stimulus combinations (stimulus_layout × flanker_direction)
and saves them for reuse during logits extraction.
"""

import os
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
from tqdm import tqdm


def generate_stimulus_images(output_dir: str, image_size: int = 128):
    """
    Generate all 28 unique stimulus combinations.
    
    Args:
        output_dir: Directory to save images
        image_size: Size of generated images
    """
    print(f"Generating {28} unique stimulus images...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the 28 combinations
    # stimulus_layout: 0-27 (28 positions in a 4x7 grid)
    # flanker_direction: 0-3 (4 directions: L/R/U/D)
    
    combinations = []
    idx = 0
    
    for stimulus_layout in range(28):
        for flanker_direction in range(4):
            # Create image
            img = Image.new('RGB', (image_size, image_size), color=(255, 255, 255))
            draw = ImageDraw.Draw(img)
            
            # Draw target bird (center)
            # Position: (x, y) where x=0-3 (columns), y=0-6 (rows)
            col = stimulus_layout % 4
            row = stimulus_layout // 4
            
            x = col * (image_size // 4) + (image_size // 8)
            y = row * (image_size // 7) + (image_size // 8)
            
            # Draw target bird (center)
            # Draw target bird (center)
            # Position: (x, y) where x=0-3 (columns), y=0-6 (rows)
            col = stimulus_layout % 4
            row = stimulus_layout // 4
            
            x = col * (image_size // 4) + (image_size // 8)
            y = row * (image_size // 7) + (image_size // 8)
            
            # Draw flanker birds (smaller) - draw all at once
            # Flanker directions: 0=L, 1=R, 2=U, 3=D
            # Flanker offsets: (-30, 0), (30, 0), (0, -30), (0, 30)
            
            # Calculate bounding box for all flankers
            flanker_boxes = []
            for i, (fx, fy) in enumerate(flanker_offsets):
                if i == flanker_direction:
                    continue
                
                # Flanker bounding box
                fx, fy = flanker_offsets[i]
                flanker_boxes.append((x+fx, y+fy, x+10, y+10))
            
            # Draw all flankers at once
            if flanker_boxes:
                draw.ellipse(flanker_boxes, fill=(150, 100, 50), outline=(0, 0, 0))
            
            # Draw target bird (larger)
            # Target bounding box
            draw.ellipse([x-20, y-20, x+20, y+20], fill=(200, 150, 50), outline=(0, 0, 0))
            
            # Save image
            filename = f"stimulus_s{stimulus_layout:02d}_f{flanker_direction}.png"
            filepath = os.path.join(output_dir, filename)
            img.save(filepath)
            
            combinations.append({
                'stimulus_layout': stimulus_layout,
                'flanker_direction': flanker_direction,
                'image_path': filepath
            })
            
            idx += 1
    
    print(f"Generated {len(combinations)} images to {output_dir}")
    
    # Save mapping
    mapping_df = pd.DataFrame(combinations)
    mapping_df.to_csv(os.path.join(output_dir, 'stimulus_mapping.csv'), index=False)
    
    return combinations


def main():
    output_dir = 'vam_data/stimulus_images'
    
    combinations = generate_stimulus_images(output_dir, image_size=128)
    
    print(f"\nStimulus mapping:")
    print(combinations.head(10))
    print(f"\nTotal: {len(combinations)} unique images")
    
    print(f"\nImages saved to {output_dir}")
    print(f"Mapping saved to {os.path.join(output_dir, 'stimulus_mapping.csv')}")


if __name__ == "__main__":
    main()
