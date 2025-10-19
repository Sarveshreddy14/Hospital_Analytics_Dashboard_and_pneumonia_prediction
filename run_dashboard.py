#!/usr/bin/env python3
"""
Hospital Analytics Dashboard Runner
Run this script to start the comprehensive Streamlit dashboard
"""

import subprocess
import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit', 'pandas', 'numpy', 'matplotlib', 
        'seaborn', 'plotly', 'tensorflow', 'pillow'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Dependencies installed successfully!")

def main():
    """Main function to run the dashboard"""
    print("🏥 Hospital Analytics Dashboard")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not Path("comprehensive_dashboard.py").exists():
        print("Error: comprehensive_dashboard.py not found!")
        print("Please run this script from the project root directory.")
        return
    
    # Check dependencies
    check_dependencies()
    
    # Run Streamlit
    print("\nStarting Streamlit dashboard...")
    print("Dashboard will be available at: http://localhost:8501")
    print("Press Ctrl+C to stop the dashboard")
    print("-" * 40)
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "comprehensive_dashboard.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    except Exception as e:
        print(f"Error running dashboard: {e}")

if __name__ == "__main__":
    main()
