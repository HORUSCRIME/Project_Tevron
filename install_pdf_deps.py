import subprocess
import sys

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"Successfully installed {package}")
        return True
    except subprocess.CalledProcessError:
        print(f"Failed to install {package}")
        return False

def main():
    print("🔧 Installing PDF generation dependencies...")
    
    packages = [
        "reportlab",  
    ]
    
    print(" Note: WeasyPrint skipped due to Windows compatibility issues")
    print("For HTML to PDF conversion, use browser Print-to-PDF feature")
    
    success_count = 0
    
    for package in packages:
        if install_package(package):
            success_count += 1
    
    print(f"\n Installation Summary:")
    print(f" Successfully installed: {success_count}/{len(packages)} packages")
    
    if success_count == len(packages):
        print(" All PDF dependencies installed successfully!")
        print(" You can now use the PDF conversion features in the dashboard.")
    else:
        print("  Some packages failed to install.")
        print(" You can still use basic PDF features with reportlab only.")
    
    print("\n Manual installation:")
    print("pip install reportlab")
    print("\n For HTML to PDF conversion alternatives:")
    print("- Use browser Print-to-PDF feature")
    print("- Install wkhtmltopdf separately (if needed)")
    print("- Use online HTML to PDF converters")

if __name__ == "__main__":
    main()