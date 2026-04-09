"""
Generate unique stimulus images for LIM task using numpy arrays.

This is a simpler, more reliable approach than PIL drawing.
"""

import os
import numpy as np
import pandas as pd


def generate_stimulus_images_numpy(output_dir: str, image_size: int = 128):
    """
    Generate all 28 unique stimulus combinations using numpy arrays.
    
    Args:
        output_dir: Directory to save images
        image_size: Size of generated images
    """
    print(f"Generating {28} unique stimulus images using numpy...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create RGB image array
    # Shape: (28, image_size, image_size, 3)
    images = np.zeros((28, image_size, image_size, 3), dtype=np.uint8)
    images[:, :, :] = 255  # White background
    
    # Define positions
    # stimulus_layout: 0-27 (28 positions in 4x7 grid)
    # flanker_direction: 0-3 (4 directions: L/R/U/D)
    
    # Draw target bird (center)
    for stimulus_layout in range(28):
        # Position: (x, y) where x=0-3 (columns), y=0-6 (rows)
        col = stimulus_layout % 4
        row = stimulus_layout // 4
        
        x_center = col * (image_size // 4) + (image_size // 8)
        y_center = row * (image_size // 7) + (image_size // 8)
        
        # Draw target bird (larger)
        # Bird size: 20x20 pixels
        x0 = x_center - 10
        y0 = y_center - 10
        x1 = x_center + 10
        y1 = y_center + 10
        
        # Fill with color (200, 150, 50) = RGB
        images[stimulus_layout, y0:y1, x0:x1] = [200, 150, 50]
        
        # Draw flanker birds (smaller)
        # Flanker directions: 0=L, 1=R, 2=U, 3=D
        # Bird size: 10x10 pixels
        # Flanker offsets: (-30, 0), (30, 0), (0, -30), (0, 30)
        flanker_offsets = [
            (-30, 0),  # L
            (30, 0),   # R
            (0, -30),   # U
            (0, 30)    # D
        ]
        
        for flanker_direction, (fx, fy) in enumerate(flanker_offsets):
            if flanker_direction == flanker_direction:
                continue
            
            # Flanker positions
            fx, fy = flanker_offsets[flanker_direction]
            
            # Draw 4 flankers
            # Top-left: (x+fx-5, y+fy-5), Top-right: (x+fx+5, y+fy-5)
            # Bottom-left: (x+fx-5, y+fy+5), Bottom-right: (x+fx+5, y+fy+5)
            
            # Top-left
            images[stimulus_layout, y0-5:y0+5, x0-5:x0+5] = [150, 100, 50]
            # Top-right
            images[stimulus_layout, y0-5:y0+5, x0+5:x0+5] = [150, 100, 50]
            # Bottom-left
            images[stimulus_layout, y0+5:y0+5, x0+5:x0+5] = [150, 100, 50]
            # Bottom-right
            images[stimulus_layout, y0+5:y0+5, x0+5:x0+5] = [150, 100, 50]
    
    print(f"Generated {28} images to {output_dir}")
    
    # Save mapping
    combinations = []
    for stimulus_layout in range(28):
        for flanker_direction in range(4):
            combinations.append({
                'stimulus_layout': stimulus_layout,
                'flanker_direction': flanker_direction,
                'image_path': f'stimulus_s{stimulus_layout:02d}_f{flanker_direction}.png'
            })
    
    mapping_df = pd.DataFrame(combinations)
    mapping_df.to_csv(os.path.join(output_dir, 'stimulus_mapping.csv'), index=False)
    
    return images, mapping_df


def main():
    output_dir = 'vam_data/stimulus_images'
    
    images, mapping_df = generate_stimulus_images_numpy(output_dir, image_size=128)
    
    # Save images
    for i, img in enumerate(images):
        filename = f'stimulus_s{i:02d}.png'
        filepath = os.path.join(output_dir, filename)
        
        # Save as PNG
        from PIL import Image
        img_pil = Image.fromarray(img)
        img_pil.save(filepath)
    
    print(f"\nStimulus mapping:")
    print(mapping_df.head(10))
    print(f"\nTotal: {len(mapping_df)} unique images")
    print(f"\nImages saved to {output_dir}")
    print(f"Mapping saved to {os.path.join(output_dir, 'stimulus_mapping.csv')}")


if __name__ == "__main__":
    main()
