import os
import sys

# Add current dir to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hwp_converter import HWPConverter

def verify():
    print("Verification script started.")
    # Look for the first HWP file in the data directory
    data_dir = r"D:\project\TodoPrj_Anti\data\rfp_files\files"
    output_dir = r"D:\project\TodoPrj_Anti\data\rfp_files"
    
    hwp_files = [f for f in os.listdir(data_dir) if f.endswith(".hwp")]
    if not hwp_files:
        print("No HWP files found in data directory.")
        return
    
    # Pick the first one
    input_file = os.path.join(data_dir, hwp_files[0])
    output_file = os.path.join(output_dir, "verification_test.pdf")
    
    print(f"Testing conversion of: {input_file}")
    print(f"Target output: {output_file}")
    
    converter = HWPConverter()
    try:
        success = converter.convert_to_pdf(input_file, output_file)
        print(f"Conversion result: {success}")
        
        if success and os.path.exists(output_file):
            print(f"✓ PDF file created successfully: {os.path.getsize(output_file)} bytes")
        else:
            print("✗ PDF file not created.")
    except Exception as e:
        print(f"✗ Exception during conversion: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify()
