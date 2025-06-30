#!/usr/bin/env python3
"""
Application runner for the Cybersecurity Drift Detection System
"""

import os
import sys
import subprocess

def check_requirements():
    """Check if all required packages are installed"""
    try:
        import flask
        import numpy
        import pandas
        import sklearn
        import plotly
        import scipy
        print("✓ All required packages are installed")
        return True
    except ImportError as e:
        print(f"✗ Missing required package: {e}")
        print("Installing requirements...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            print("✓ Requirements installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("✗ Failed to install requirements")
            return False

def setup_directories():
    """Create necessary directories"""
    directories = ['data', 'models', 'static/css', 'static/js', 'templates']
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Directory '{directory}' ready")

def main():
    """Main application runner"""
    print("🔒 Cybersecurity Drift Detection System")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("✗ Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Setup directories
    setup_directories()
    
    # Import and run the Flask app
    try:
        from app import app, flask_config
        print("\n🚀 Starting the application...")
        print(f"📊 Dashboard will be available at: http://localhost:{flask_config.PORT}")
        print("🔧 Press Ctrl+C to stop the application")
        print("-" * 50)
        
        app.run(
            host=flask_config.HOST,
            port=flask_config.PORT,
            debug=flask_config.DEBUG
        )
        
    except KeyboardInterrupt:
        print("\n\n🛑 Application stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting application: {e}")
        print("\nTroubleshooting tips:")
        print("1. Ensure all files are in the correct directory structure")
        print("2. Check that port 5000 is not already in use")
        print("3. Verify all Python dependencies are installed")
        sys.exit(1)

if __name__ == "__main__":
    main()