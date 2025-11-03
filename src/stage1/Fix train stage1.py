"""
Automatic fix script for train_stage1_moco.py
Replaces mode='unpaired' with paired=False
"""

import sys
import os

def fix_file(filepath):
    """Fix the train_stage1_moco.py file"""
    
    # Read the file
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if already fixed
    if "paired=False" in content and "mode='unpaired'" not in content:
        print("✅ File is already fixed!")
        return True
    
    # Backup original
    backup_path = filepath + '.backup'
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"📦 Backup created: {backup_path}")
    
    # Apply fixes
    original_content = content
    
    # Fix 1: Replace mode='unpaired' with paired=False
    content = content.replace(
        "        mode='unpaired',  # Only use low-light images",
        "        paired=False,  # Only use low-light images (unpaired mode)"
    )
    
    # Fix 2: Add normalize=True if not present
    if "train_dataset = LowLightDataset(" in content:
        # Find the section
        lines = content.split('\n')
        new_lines = []
        in_dataset_creation = False
        added_normalize = False
        
        for i, line in enumerate(lines):
            new_lines.append(line)
            
            if "train_dataset = LowLightDataset(" in line:
                in_dataset_creation = True
            
            if in_dataset_creation and "augment=True" in line and not added_normalize:
                # Check if next line closes the parenthesis
                if i + 1 < len(lines) and ')' in lines[i + 1]:
                    # Add normalize before closing
                    new_lines.append("        normalize=True")
                    added_normalize = True
        
        content = '\n'.join(new_lines)
    
    # Write fixed content
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    if content != original_content:
        print("✅ File fixed successfully!")
        print("\nChanges made:")
        print("  - Changed: mode='unpaired' → paired=False")
        print("  - Added: normalize=True parameter")
        return True
    else:
        print("⚠️  No changes needed or pattern not found")
        return False

if __name__ == '__main__':
    # Default path
    filepath = r"D:\projects\SemRoCL\src\train_stage1_moco.py"
    
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        print("\nUsage:")
        print("  python fix_train_stage1.py")
        print("  python fix_train_stage1.py <path_to_file>")
        sys.exit(1)
    
    print(f"🔧 Fixing: {filepath}\n")
    
    if fix_file(filepath):
        print("\n✅ Done! You can now run training:")
        print("   python src/train_stage1_moco.py --config configs/train_stage1.yaml")
    else:
        print("\n❌ Fix failed. Please apply manual fix.")