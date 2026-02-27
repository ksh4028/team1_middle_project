import os
import sys

# Add the directory to sys.path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from hwp_converter import HWPConverter
    print("✓ Successfully imported HWPConverter")
    
    converter = HWPConverter()
    print("✓ Successfully initialized HWPConverter")
    
    # Check if we can get the HWP object (this will fail if HWP is not installed)
    hwp = converter._get_hwp_object()
    if hwp:
        print("✓ Successfully connected to HWP COM object")
        hwp.Quit()
    else:
        print("✗ Failed to connect to HWP COM object (Is Hancom Office installed?)")
        
except Exception as e:
    print(f"✗ Test failed: {e}")
